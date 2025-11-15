#!/usr/bin/env python3
"""
Teacher-Student Models with Message Passing for Emergent Language
PRODUCTION VERSION - Full integration with message_channel.py components
Extends base models (RNN/Transformer) with communication capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import production message channel components
from .message_channel import (
    GumbelSoftmaxMessageChannel,
    ContinuousMessageChannel,
    HybridMessageChannel,
    MessageDecoder
)


class TeacherWithMessageEncoder(nn.Module):
    """
    Teacher model that generates messages to help student learn
    
    Architecture:
        Input -> Base Model -> Prediction
                  |
                  └-> Message Encoder -> Message
    """
    
    def __init__(self, base_model, message_channel, fusion_type='concat'):
        """
        Args:
            base_model: Pre-initialized RNN/Transformer model
            message_channel: Message channel (discrete/continuous/hybrid)
            fusion_type: How to extract features ('concat', 'pool', 'last')
        """
        super().__init__()
        self.base_model = base_model
        self.message_channel = message_channel
        self.fusion_type = fusion_type
        
        # Project hidden state to message channel input dimension
        self.hidden_to_msg = nn.Linear(
            base_model.nhid, 
            message_channel.hidden_dim if hasattr(message_channel, 'hidden_dim') else base_model.nhid
        )
        
    def extract_features(self, output, hidden):
        """
        Extract features from model output for message generation
        
        Args:
            output: Model output logits (seq_len, batch, vocab_size)
            hidden: Hidden state (varies by model type)
            
        Returns:
            features: (batch, hidden_dim)
        """
        if isinstance(hidden, tuple):  # LSTM
            # Use last layer's hidden state
            h = hidden[0][-1]  # (batch, hidden_dim)
        else:  # GRU or other
            h = hidden[-1]
        
        return h
    
    def forward(self, input, hidden=None, return_message=True):
        """
        Forward pass with message generation
        
        Args:
            input: Input tokens (seq_len, batch)
            hidden: Initial hidden state
            return_message: Whether to generate message
            
        Returns:
            output: Prediction logits (seq_len, batch, vocab_size)
            message: Encoded message (if return_message=True)
            msg_logits: Message logits (for discrete channels)
            hidden: Updated hidden state
        """
        # Get teacher's prediction
        output, hidden = self.base_model(input, hidden)
        
        if not return_message:
            return output, None, None, hidden
        
        # Extract features for message
        features = self.extract_features(output, hidden)
        
        # Project to message dimension
        msg_features = self.hidden_to_msg(features)
        
        # Encode message
        if hasattr(self.message_channel, 'encode'):
            if isinstance(self.message_channel, 
                         (GumbelSoftmaxMessageChannel, HybridMessageChannel)):
                # Discrete or hybrid
                result = self.message_channel.encode(msg_features)
                if isinstance(result, tuple):
                    message, msg_logits_or_indices = result
                else:
                    message = result
                    msg_logits_or_indices = None
            else:
                # Continuous
                message = self.message_channel.encode(msg_features)
                msg_logits_or_indices = None
        else:
            raise ValueError("Message channel must have encode() method")
        
        return output, message, msg_logits_or_indices, hidden


class StudentWithMessageDecoder(nn.Module):
    """
    Student model that uses teacher's messages to improve predictions
    PRODUCTION VERSION - Uses MessageDecoder from message_channel.py
    
    Architecture:
        Input -> Base Model -> Output
                      ↑
        Message -> MessageDecoder -> Fused Hidden
    """
    
    def __init__(self, base_model, message_channel, message_decoder=None, fusion_method='add'):
        """
        Args:
            base_model: Pre-initialized RNN/Transformer model
            message_channel: Message channel (same as teacher's)
            message_decoder: MessageDecoder instance (if None, creates one)
            fusion_method: How to fuse message ('add', 'concat', 'attention')
        """
        super().__init__()
        self.base_model = base_model
        self.message_channel = message_channel
        self.fusion_method = fusion_method
        
        # Use production MessageDecoder from message_channel.py
        if message_decoder is None:
            if hasattr(message_channel, 'vocab_size'):
                # Discrete message channel
                self.message_decoder = MessageDecoder(
                    vocab_size=message_channel.vocab_size,
                    msg_length=message_channel.msg_length,
                    hidden_dim=base_model.nhid,
                    fusion_type=fusion_method
                )
            else:
                # Continuous/Hybrid - adapt vocab_size parameter
                # For continuous, treat msg_dim as vocab_size
                self.message_decoder = MessageDecoder(
                    vocab_size=message_channel.msg_dim if hasattr(message_channel, 'msg_dim') else 64,
                    msg_length=message_channel.msg_length,
                    hidden_dim=base_model.nhid,
                    fusion_type=fusion_method
                )
        else:
            self.message_decoder = message_decoder
        
        # Output projection (if needed for dimension mismatch)
        self.output_proj = None
        if fusion_method in ['concat', 'attention']:
            # May need to project back to vocab_size
            vocab_size = base_model.decoder.out_features if hasattr(base_model, 'decoder') else None
            if vocab_size:
                self.output_proj = nn.Linear(base_model.nhid, vocab_size)
    
    def forward(self, input, message, hidden=None):
        """
        Forward pass using teacher's message with production MessageDecoder
        
        Args:
            input: Input tokens (seq_len, batch)
            message: Message from teacher (batch, msg_length, *)
            hidden: Initial hidden state
            
        Returns:
            output: Prediction logits (seq_len, batch, vocab_size)
            hidden: Updated hidden state
        """
        seq_len, batch_size = input.size()
        
        # Get student's hidden states from embedding + RNN
        # We need to intercept before the final decoder layer
        if hasattr(self.base_model, 'encoder') and hasattr(self.base_model, 'rnn'):
            # Standard RNN model structure
            emb = self.base_model.encoder(input)
            if hasattr(self.base_model, 'drop'):
                emb = self.base_model.drop(emb)
            
            rnn_output, hidden = self.base_model.rnn(emb, hidden)
            
            if hasattr(self.base_model, 'drop'):
                rnn_output = self.base_model.drop(rnn_output)
            
            # Decode message and fuse with RNN output
            if message is not None:
                # rnn_output: (seq_len, batch, hidden_dim)
                # Transpose for MessageDecoder: (batch, seq_len, hidden_dim)
                rnn_output_T = rnn_output.transpose(0, 1)
                
                # Use production MessageDecoder with fusion
                fused_output_T = self.message_decoder(message, rnn_output_T)
                
                # Transpose back: (seq_len, batch, hidden_dim)
                fused_output = fused_output_T.transpose(0, 1)
            else:
                fused_output = rnn_output
            
            # Final decoder layer
            if hasattr(self.base_model, 'decoder'):
                output = self.base_model.decoder(fused_output)
            elif self.output_proj:
                output = self.output_proj(fused_output)
            else:
                output = fused_output
        else:
            # Fallback: use base model as-is
            output, hidden = self.base_model(input, hidden)
            
            if message is not None:
                # Simple fusion at output level
                seq_len, batch, vocab_size = output.size()
                output_T = output.transpose(0, 1)  # (batch, seq_len, vocab_size)
                
                # Decode message (without student hidden)
                msg_decoded = self.message_decoder(message)  # (batch, hidden_dim)
                
                # Broadcast and add
                msg_broadcast = msg_decoded.unsqueeze(1).expand(batch, seq_len, msg_decoded.size(-1))
                
                # Need to project if dimensions don't match
                if msg_broadcast.size(-1) != vocab_size:
                    if self.output_proj is None:
                        self.output_proj = nn.Linear(msg_broadcast.size(-1), vocab_size).to(output.device)
                    msg_broadcast = self.output_proj(msg_broadcast)
                
                output_T = output_T + msg_broadcast
                output = output_T.transpose(0, 1)
        
        return output, hidden


def test_emergent_language_models():
    """Test teacher-student models with message passing"""
    print("Testing Emergent Language Models...")
    print("="*80)
    
    # Mock RNN model for testing
    class MockRNN(nn.Module):
        def __init__(self, vocab_size=10000, emsize=650, nhid=650, nlayers=2):
            super().__init__()
            self.nhid = nhid
            self.nlayers = nlayers
            self.encoder = nn.Embedding(vocab_size, emsize)
            self.rnn = nn.LSTM(emsize, nhid, nlayers, dropout=0.5)
            self.decoder = nn.Linear(nhid, vocab_size)
            
        def init_hidden(self, batch_size):
            weight = next(self.parameters())
            return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                   weight.new_zeros(self.nlayers, batch_size, self.nhid))
        
        def forward(self, input, hidden):
            emb = self.encoder(input)
            output, hidden = self.rnn(emb, hidden)
            decoded = self.decoder(output)
            return decoded, hidden
    
    # Create models
    vocab_size = 10000
    batch_size = 4
    seq_len = 10
    
    teacher_base = MockRNN(vocab_size=vocab_size)
    student_base = MockRNN(vocab_size=vocab_size)
    
    # Test with discrete messages
    print("\n1. Discrete Message Channel")
    print("-"*80)
    msg_channel_discrete = GumbelSoftmaxMessageChannel(
        vocab_size=50, msg_length=10, hidden_dim=650
    )
    teacher_discrete = TeacherWithMessageEncoder(teacher_base, msg_channel_discrete)
    student_discrete = StudentWithMessageDecoder(student_base, msg_channel_discrete)
    
    # Forward pass
    input_tokens = torch.randint(0, vocab_size, (seq_len, batch_size))
    teacher_hidden = teacher_base.init_hidden(batch_size)
    student_hidden = student_base.init_hidden(batch_size)
    
    teacher_output, message, msg_logits, teacher_hidden = teacher_discrete(
        input_tokens, teacher_hidden
    )
    print(f"Teacher output shape: {teacher_output.shape}")  # (10, 4, 10000)
    print(f"Message shape: {message.shape}")  # (4, 10, 50)
    
    student_output, student_hidden = student_discrete(
        input_tokens, message, student_hidden
    )
    print(f"Student output shape: {student_output.shape}")  # (10, 4, 10000)
    
    # Test with continuous messages
    print("\n2. Continuous Message Channel")
    print("-"*80)
    teacher_base_2 = MockRNN(vocab_size=vocab_size)
    student_base_2 = MockRNN(vocab_size=vocab_size)
    
    msg_channel_continuous = ContinuousMessageChannel(
        msg_dim=64, msg_length=10, hidden_dim=650
    )
    teacher_continuous = TeacherWithMessageEncoder(teacher_base_2, msg_channel_continuous)
    student_continuous = StudentWithMessageDecoder(student_base_2, msg_channel_continuous)
    
    teacher_hidden_2 = teacher_base_2.init_hidden(batch_size)
    student_hidden_2 = student_base_2.init_hidden(batch_size)
    
    teacher_output_2, message_2, _, teacher_hidden_2 = teacher_continuous(
        input_tokens, teacher_hidden_2
    )
    print(f"Teacher output shape: {teacher_output_2.shape}")  # (10, 4, 10000)
    print(f"Message shape: {message_2.shape}")  # (4, 10, 64)
    
    student_output_2, student_hidden_2 = student_continuous(
        input_tokens, message_2, student_hidden_2
    )
    print(f"Student output shape: {student_output_2.shape}")  # (10, 4, 10000)
    
    print("\n" + "="*80)
    print("All tests passed! ✓")


if __name__ == '__main__':
    test_emergent_language_models()
