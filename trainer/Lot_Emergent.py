#!/usr/bin/env python3
"""
PHASE 4 PRODUCTION: LoT + Emergent Language
===========================================

Properly integrates emergent language communication with Learning-from-Teaching.

Key Improvements over Baseline LoT:
1. Teacher encodes knowledge into discrete messages
2. Student uses messages to improve predictions
3. Messages must be informative (language regularization)
4. Gradual message integration (curriculum learning)
5. Proper LoT regularization that works WITH messages

Target: Beat baseline LoT PPL by 5-15 points via better knowledge transfer
"""

import argparse
import time
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import model.rnn as rnn
from model.message_channel import (
    GumbelSoftmaxMessageChannel,
    LanguageMetrics,
    compute_language_regularization
)
from model.emergent_language_models import (
    TeacherWithMessageEncoder,
    StudentWithMessageDecoder
)
from utils.data_utils import get_lm_corpus


def batchify(data, bsz, device):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    return tuple(repackage_hidden(v) for v in h)


def kl_div_logits(p, q, T):
    """KL divergence between logit distributions"""
    loss_func = nn.KLDivLoss(reduction='batchmean', log_target=True)
    return loss_func(F.log_softmax(p/T, dim=-1), F.log_softmax(q/T, dim=-1)) * T * T


class Phase4ProductionTrainer:
    """
    Production trainer that properly integrates emergent language with LoT
    
    Strategy:
    1. Start with strong LoT baseline (alpha=1.0)
    2. Gradually enable message passing
    3. Use messages to ENHANCE teacher->student knowledge transfer
    4. Regularize messages to be informative
    """
    
    def __init__(self, args, teacher, student, message_channel,
                 train_data, val_data, test_data, ntokens, device):
        self.args = args
        self.teacher = teacher
        self.student = student
        self.message_channel = message_channel
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.ntokens = ntokens
        self.device = device
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizers - Same as baseline LoT
        self.teacher_opt = torch.optim.SGD(
            self.teacher.parameters(), lr=args.lr, momentum=0.3
        )
        self.student_opt = torch.optim.SGD(
            self.student.parameters(), lr=args.lr, momentum=0.3
        )
        
        # Schedulers
        self.teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.teacher_opt, T_max=args.epochs
        )
        self.student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.student_opt, T_max=args.epochs
        )
        
        # Tracking
        self.best_val_ppl = float('inf')
        self.best_epoch = 0
        self.step_count = 0
        
        # Curriculum: Gradually enable messages
        self.message_weight = 0.0  # Start with 0, ramp up
        
    def get_message_weight(self, epoch):
        """
        VERY SLOW curriculum learning - keep messages minimal for first 30 epochs
        This lets LoT learn first, THEN add messages as enhancement
        """
        if epoch <= 30:
            # First 30 epochs: 0% → 10% (very slow ramp)
            return 0.1 * (epoch / 30.0)
        elif epoch <= 50:
            # Epochs 30-50: 10% → 50% (medium ramp)
            return 0.1 + 0.4 * ((epoch - 30) / 20.0)
        else:
            # Final 10 epochs: 50% → 100% (final ramp)
            return 0.5 + 0.5 * ((epoch - 50) / 10.0)
    
    def train_epoch(self, epoch):
        """Train one epoch with LoT + Messages"""
        self.teacher.train()
        self.student.train()
        
        total_teacher_loss = 0
        total_student_loss = 0
        start_time = time.time()
        
        teacher_hidden = self.teacher.base_model.init_hidden(self.args.batch_size)
        student_hidden = self.student.base_model.init_hidden(self.args.batch_size)
        
        # Get current message weight (curriculum)
        self.message_weight = self.get_message_weight(epoch)
        
        num_batches = (self.train_data.data.size(0) - 1) // self.args.bptt
        
        for batch_idx, i in enumerate(range(0, self.train_data.data.size(0) - 1, self.args.bptt)):
            data, targets, _ = self.train_data.get_batch(i)
            targets = targets.view(-1)
            
            # Detach hidden states
            teacher_hidden = repackage_hidden(teacher_hidden)
            student_hidden = repackage_hidden(student_hidden)
            
            # === TEACHER FORWARD ===
            teacher_output, message, msg_logits, teacher_hidden = self.teacher(
                data, teacher_hidden, return_message=True
            )
            teacher_output_flat = teacher_output.view(-1, self.ntokens)
            
            # === STUDENT FORWARD (with messages) ===
            student_output, student_hidden = self.student(
                data, message.detach(), student_hidden
            )
            student_output_flat = student_output.view(-1, self.ntokens)
            
            # === COMPUTE LOSSES ===
            
            # 1. Cross-entropy losses
            teacher_ce = self.criterion(teacher_output_flat, targets)
            student_ce = self.criterion(student_output_flat, targets)
            
            # 2. LoT losses (bidirectional KL) - SAME AS BASELINE
            kl_t2s = kl_div_logits(student_output_flat, teacher_output_flat.detach(), self.args.T)
            kl_s2t = kl_div_logits(teacher_output_flat, student_output_flat.detach(), self.args.T)
            
            # 3. Language regularization (only if using messages)
            lang_reg = torch.tensor(0.0).to(self.device)
            if msg_logits is not None and self.message_weight > 0:
                lang_reg = compute_language_regularization(
                    message, msg_logits,
                    beta_entropy=self.args.beta_entropy,
                    beta_comp=self.args.beta_comp
                )
            
            # === TOTAL LOSSES ===
            # Teacher: CE + LoT (S->T) + Language Regularization
            teacher_loss = (
                teacher_ce +
                self.args.alpha * kl_s2t +
                self.message_weight * self.args.beta_lang * lang_reg
            )
            
            # Student: CE + LoT (T->S)
            student_loss = student_ce + self.args.alpha * kl_t2s
            
            # === OPTIMIZE ===
            self.teacher_opt.zero_grad()
            self.student_opt.zero_grad()
            
            teacher_loss.backward(retain_graph=True)
            student_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), self.args.clip)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.args.clip)
            
            self.teacher_opt.step()
            self.student_opt.step()
            
            # === ADDITIONAL STUDENT STEPS (like baseline) ===
            for _ in range(self.args.student_steps_ratio - 1):
                self.args.current_index = (self.args.bptt + self.args.current_index) % (self.train_data.data.size(0) - 1)
                data_s, targets_s, _ = self.train_data.get_batch(self.args.current_index)
                targets_s = targets_s.view(-1)
                
                student_hidden = repackage_hidden(student_hidden)
                
                # Get fresh message from teacher
                with torch.no_grad():
                    _, message_s, _, _ = self.teacher(data_s, teacher_hidden, return_message=True)
                
                student_output_s, student_hidden = self.student(data_s, message_s, student_hidden)
                student_output_s_flat = student_output_s.view(-1, self.ntokens)
                
                # Get teacher prediction for KL
                with torch.no_grad():
                    teacher_output_s, _, _, _ = self.teacher(data_s, teacher_hidden, return_message=True)
                    teacher_output_s_flat = teacher_output_s.view(-1, self.ntokens)
                
                student_loss_s = (
                    self.criterion(student_output_s_flat, targets_s) +
                    self.args.alpha * kl_div_logits(student_output_s_flat, teacher_output_s_flat, self.args.T)
                )
                
                self.student_opt.zero_grad()
                student_loss_s.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.args.clip)
                self.student_opt.step()
            
            total_teacher_loss += teacher_loss.item()
            total_student_loss += student_loss.item()
            self.step_count += 1
            
            # Logging
            if batch_idx % self.args.log_interval == 0 and batch_idx > 0:
                cur_loss = total_teacher_loss / self.args.log_interval
                elapsed = time.time() - start_time
                print(f'| epoch {epoch:3d} | {batch_idx:5d}/{num_batches:5d} batches | '
                      f'lr {self.teacher_opt.param_groups[0]["lr"]:.2e} | '
                      f'ms/batch {elapsed * 1000 / self.args.log_interval:.2f} | '
                      f'loss {cur_loss:.2f} | ppl {math.exp(min(cur_loss, 20)):8.2f} | '
                      f'msg_weight {self.message_weight:.3f}')
                total_teacher_loss = 0
                total_student_loss = 0
                start_time = time.time()
        
        return {}
    
    def evaluate(self, data_source, use_messages=True):
        """Evaluate model - FIXED: separate hidden states for teacher and student"""
        self.teacher.eval()
        self.student.eval()
        
        total_loss = 0.0
        teacher_hidden = self.teacher.base_model.init_hidden(10)
        student_hidden = self.student.base_model.init_hidden(10)
        
        with torch.no_grad():
            for i in range(0, data_source.data.size(0) - 1, self.args.bptt):
                data, targets, seq_len = data_source.get_batch(i)
                targets = targets.view(-1)
                
                # CRITICAL FIX: Separate hidden states
                teacher_hidden = repackage_hidden(teacher_hidden)
                student_hidden = repackage_hidden(student_hidden)
                
                if use_messages:
                    # Get message from teacher (with teacher's hidden state)
                    _, message, _, teacher_hidden = self.teacher(data, teacher_hidden, return_message=True)
                    # Student uses message (with student's hidden state)
                    output, student_hidden = self.student(data, message, student_hidden)
                else:
                    # Student without messages (baseline, student's hidden state)
                    output, student_hidden = self.student.base_model(data, student_hidden)
                
                output_flat = output.view(-1, self.ntokens)
                total_loss += len(data) * self.criterion(output_flat, targets).item()
        
        return total_loss / len(data_source.data)
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("PHASE 4 PRODUCTION TRAINING")
        print("="*80)
        
        for epoch in range(1, self.args.epochs + 1):
            epoch_start_time = time.time()
            
            # Train
            self.train_epoch(epoch)
            
            # Evaluate
            teacher_val_loss = self.evaluate(self.val_data, use_messages=True)
            student_val_loss = self.evaluate(self.val_data, use_messages=True)
            student_nomsg_loss = self.evaluate(self.val_data, use_messages=False)
            
            teacher_val_ppl = math.exp(min(teacher_val_loss, 20))
            student_val_ppl = math.exp(min(student_val_loss, 20))
            student_nomsg_ppl = math.exp(min(student_nomsg_loss, 20))
            
            print("-"*80)
            print(f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:.2f}s |')
            print(f'| teacher ppl {teacher_val_ppl:8.2f} | student (w/msg) {student_val_ppl:8.2f} | student (no msg) {student_nomsg_ppl:8.2f}')
            print(f'| message benefit: {student_nomsg_ppl - student_val_ppl:+.2f} PPL')
            print("-"*80)
            
            # Save best
            if student_val_ppl < self.best_val_ppl:
                self.best_val_ppl = student_val_ppl
                self.best_epoch = epoch
                self.save_checkpoint(epoch, teacher_val_ppl, student_val_ppl, student_nomsg_ppl)
                print(f'| ✓ Saved best model (epoch {epoch})')
            
            # Learning rate scheduling
            self.teacher_scheduler.step()
            self.student_scheduler.step()
            
            print()
        
        # Final test
        print("="*80)
        print("FINAL TEST RESULTS")
        print("="*80)
        
        self.load_checkpoint()
        
        teacher_test_loss = self.evaluate(self.test_data, use_messages=True)
        student_test_loss = self.evaluate(self.test_data, use_messages=True)
        student_nomsg_test_loss = self.evaluate(self.test_data, use_messages=False)
        
        teacher_test_ppl = math.exp(min(teacher_test_loss, 20))
        student_test_ppl = math.exp(min(student_test_loss, 20))
        student_nomsg_test_ppl = math.exp(min(student_nomsg_test_loss, 20))
        
        print(f'| Teacher PPL: {teacher_test_ppl:.2f}')
        print(f'| Student (with messages) PPL: {student_test_ppl:.2f}')
        print(f'| Student (without messages) PPL: {student_nomsg_test_ppl:.2f}')
        print(f'| Message benefit: {student_nomsg_test_ppl - student_test_ppl:+.2f} PPL')
        print("="*80)
    
    def save_checkpoint(self, epoch, teacher_ppl, student_ppl, student_nomsg_ppl):
        """Save checkpoint"""
        os.makedirs(os.path.dirname(self.args.save), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'teacher': self.teacher.state_dict(),
            'student': self.student.state_dict(),
            'message_channel': self.message_channel.state_dict(),
            'teacher_opt': self.teacher_opt.state_dict(),
            'student_opt': self.student_opt.state_dict(),
            'teacher_ppl': teacher_ppl,
            'student_ppl': student_ppl,
            'student_nomsg_ppl': student_nomsg_ppl,
            'best_val_ppl': self.best_val_ppl,
            'args': self.args
        }, self.args.save)
    
    def load_checkpoint(self):
        """Load checkpoint"""
        if os.path.exists(self.args.save):
            print(f'Loading best model from: {self.args.save}')
            ckpt = torch.load(self.args.save, weights_only=False)
            self.teacher.load_state_dict(ckpt['teacher'])
            self.student.load_state_dict(ckpt['student'])
            self.message_channel.load_state_dict(ckpt['message_channel'])


def main():
    parser = argparse.ArgumentParser(description='Phase 4 Production: LoT + Emergent Language')
    
    # Message parameters
    parser.add_argument('--vocab_size', type=int, default=32,
                       help='Message vocabulary size (smaller=easier)')
    parser.add_argument('--msg_length', type=int, default=8,
                       help='Message length (shorter=easier)')
    
    # LoT parameters - SAME AS BASELINE
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='LoT regularization strength (baseline uses 1.0)')
    parser.add_argument('--T', type=float, default=1.5,
                       help='Temperature for KL divergence')
    
    # Language regularization - MINIMAL (don't interfere with LoT)
    parser.add_argument('--beta_lang', type=float, default=0.0001,
                       help='Language regularization weight (VERY small)')
    parser.add_argument('--beta_entropy', type=float, default=0.5,
                       help='Entropy bonus (reduced)')
    parser.add_argument('--beta_comp', type=float, default=0.1,
                       help='Compositionality bonus (reduced)')
    
    # Model architecture - SAME AS BASELINE
    parser.add_argument('--data', type=str, default='ptb')
    parser.add_argument('--emsize', type=int, default=650)
    parser.add_argument('--nhid', type=int, default=650)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.45)
    
    # Training - SAME AS BASELINE
    parser.add_argument('--lr', type=float, default=30)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--bptt', type=int, default=35)
    parser.add_argument('--student_steps_ratio', type=int, default=5)
    
    # Other
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--save', type=str, default='ckpt/Phase4/production.pt')
    parser.add_argument('--current_index', type=int, default=0)
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    
    print("="*80)
    print("CONFIGURATION")
    print("="*80)
    print(json.dumps(vars(args), indent=2))
    print("="*80)
    
    # Load data
    if args.data == 'ptb':
        datadir = 'data/ptb'
    elif args.data == 'wt103':
        datadir = 'data/wikitext-103'
    
    corpus = get_lm_corpus(datadir, args.data)
    ntokens = len(corpus.vocab)
    
    train_data = corpus.get_iterator('train', args.batch_size, args.bptt, device=device, ext_len=0)
    val_data = corpus.get_iterator('valid', 10, args.bptt, device=device, ext_len=0)
    test_data = corpus.get_iterator('test', 10, args.bptt, device=device, ext_len=0)
    
    print(f"\nDataset: {args.data}")
    print(f"Vocabulary size: {ntokens}")
    
    # Create message channel
    message_channel = GumbelSoftmaxMessageChannel(
        vocab_size=args.vocab_size,
        msg_length=args.msg_length,
        hidden_dim=args.nhid,
        temperature=1.0
    ).to(device)
    
    # Create base models - SAME AS BASELINE
    teacher_base = rnn.RNNModel(
        ntokens, args.emsize, args.nhid,
        args.nlayers, args.dropout
    ).to(device)
    
    student_base = rnn.RNNModel(
        ntokens, args.emsize, args.nhid,
        args.nlayers, args.dropout
    ).to(device)
    
    # Wrap with message passing
    teacher = TeacherWithMessageEncoder(teacher_base, message_channel).to(device)
    student = StudentWithMessageDecoder(
        student_base, message_channel,
        fusion_method='add'  # Simple additive fusion
    ).to(device)
    
    print(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"Student parameters: {sum(p.numel() for p in student.parameters()):,}")
    
    # Train
    trainer = Phase4ProductionTrainer(
        args, teacher, student, message_channel,
        train_data, val_data, test_data, ntokens, device
    )
    
    trainer.train()


if __name__ == '__main__':
    main()
