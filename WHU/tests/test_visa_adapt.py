"""CPU contract checks; no WHU-MARS data or pretrained weights are required."""
import unittest
import tempfile
from pathlib import Path

import torch

from model.backbones.vit_pytorch import TransReID
from model.visa_adapt import DLFM, ETGM, make_retrieval_feature
from solver.scheduler_factory import create_scheduler
from solver.make_optimizer import make_optimizer
from config import cfg


class VisaAdaptTest(unittest.TestCase):
    def test_experiment_configs_keep_uad_training_recipe(self):
        folder = Path(__file__).resolve().parents[1] / 'configs'
        configurations = []
        for name in ('UAD.yml', 'UAD_T0.yml', 'UAD_ViSA.yml'):
            option = cfg.clone()
            option.merge_from_file(str(folder / name))
            configurations.append(option)
        baseline, token, full = configurations
        for adapted in (token, full):
            self.assertEqual(adapted.SOLVER, baseline.SOLVER)
            self.assertEqual(adapted.DATALOADER, baseline.DATALOADER)
            self.assertEqual(adapted.INPUT, baseline.INPUT)
            self.assertEqual(adapted.DATASETS, baseline.DATASETS)
            for key in ('PRETRAIN_PATH', 'METRIC_LOSS_TYPE', 'PCA_LOSS_WEIGHT',
                        'GPD_MOMENTUM', 'NO_MARGIN'):
                self.assertEqual(getattr(adapted.MODEL, key), getattr(baseline.MODEL, key))
        self.assertEqual(baseline.TEST.IMS_PER_BATCH, 512)
        self.assertEqual(token.TEST.IMS_PER_BATCH, 512)
        self.assertEqual(full.TEST.IMS_PER_BATCH, 32)

    def test_readme_four_gpu_overrides_reach_scheduler(self):
        option = cfg.clone()
        option.merge_from_file(str(Path(__file__).resolve().parents[1] /
                                   'configs' / 'UAD_ViSA.yml'))
        option.merge_from_list(['MODEL.DIST_TRAIN', 'True',
                                'SOLVER.IMS_PER_BATCH', '256',
                                'SOLVER.BASE_LR', '0.032'])
        self.assertTrue(option.MODEL.DIST_TRAIN)
        self.assertEqual(option.SOLVER.IMS_PER_BATCH, 256)
        self.assertEqual(option.SOLVER.BASE_LR, 0.032)
        old = torch.nn.Parameter(torch.ones(()))
        new = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([
            {'params': [old], 'lr': 0.032, 'visa_new': False},
            {'params': [new], 'lr': 0.064, 'visa_new': True},
        ])
        scheduler = create_scheduler(option, optimizer)
        for epoch in (1, 5, 60, 120):
            scheduler.step(epoch)
            old_lr, new_lr = (group['lr'] for group in optimizer.param_groups)
            self.assertAlmostEqual(new_lr, 2 * old_lr, places=8)

    def test_optimizer_new_parameters_only(self):
        config = cfg.clone()
        config.defrost()
        config.MODEL.VISA_ADAPT.MODE = 'full'
        config.SOLVER.OPTIMIZER_NAME = 'SGD'
        config.SOLVER.BASE_LR = 0.008
        config.SOLVER.BIAS_LR_FACTOR = 2
        model = torch.nn.Module()
        model.base = torch.nn.Module()
        model.base.old_weight = torch.nn.Parameter(torch.ones(1))
        model.base.specific_pos_embed = torch.nn.Parameter(torch.ones(1))
        model.etgm = torch.nn.Linear(1, 1)
        optimizer, _ = make_optimizer(config, model, torch.nn.Linear(1, 1))
        rates = {name: next(group['lr'] for group in optimizer.param_groups
                            if parameter is group['params'][0])
                 for name, parameter in model.named_parameters()}
        self.assertEqual(rates['base.old_weight'], 0.008)
        self.assertEqual(rates['base.specific_pos_embed'], 0.016)
        self.assertEqual(rates['etgm.weight'], 0.016)
        self.assertEqual(rates['etgm.bias'], 0.032)
        flags = {name: next(group['visa_new'] for group in optimizer.param_groups
                            if parameter is group['params'][0])
                 for name, parameter in model.named_parameters()}
        self.assertFalse(flags['base.old_weight'])
        self.assertTrue(flags['etgm.bias'])

        # The README's four-GPU command overrides the single-GPU YAML LR.
        config.SOLVER.BASE_LR = 0.032
        ddp_optimizer, _ = make_optimizer(config, model, torch.nn.Linear(1, 1))
        ddp_rates = {name: next(group['lr'] for group in ddp_optimizer.param_groups
                                if parameter is group['params'][0])
                     for name, parameter in model.named_parameters()}
        self.assertEqual(ddp_rates['base.old_weight'], 0.032)
        self.assertEqual(ddp_rates['base.specific_pos_embed'], 0.064)
        self.assertEqual(ddp_rates['etgm.weight'], 0.064)
        self.assertEqual(ddp_rates['etgm.bias'], 0.128)

    def test_lr_multiplier_survives_warmup_and_cosine(self):
        config = cfg.clone()
        config.defrost()
        config.MODEL.VISA_ADAPT.MODE = 'full'
        config.SOLVER.BASE_LR = 0.008
        config.SOLVER.MAX_EPOCHS = 20
        config.SOLVER.WARMUP_EPOCHS = 5
        parameters = [torch.nn.Parameter(torch.ones(())) for _ in range(4)]
        optimizer = torch.optim.SGD([
            {'params': [parameters[0]], 'lr': 0.008, 'visa_new': False},
            {'params': [parameters[1]], 'lr': 0.016, 'visa_new': True},
            {'params': [parameters[2]], 'lr': 0.016, 'visa_new': False},
            {'params': [parameters[3]], 'lr': 0.032, 'visa_new': True},
        ])
        scheduler = create_scheduler(config, optimizer)
        baseline_config = config.clone()
        baseline_config.MODEL.VISA_ADAPT.MODE = 'off'
        baseline_optimizer = torch.optim.SGD([
            {'params': [torch.nn.Parameter(torch.ones(()))], 'lr': 0.008},
            {'params': [torch.nn.Parameter(torch.ones(()))], 'lr': 0.016},
        ])
        baseline_scheduler = create_scheduler(baseline_config, baseline_optimizer)
        for epoch in (0, 1, 5, 10, 19):
            scheduler.step(epoch)
            baseline_scheduler.step(epoch)
            rates = [group['lr'] for group in optimizer.param_groups]
            baseline_rates = [group['lr'] for group in baseline_optimizer.param_groups]
            self.assertAlmostEqual(rates[1] / rates[0], 2., places=6)
            self.assertAlmostEqual(rates[3] / rates[2], 2., places=6)
            self.assertAlmostEqual(rates[0], baseline_rates[0], places=8)
            self.assertAlmostEqual(rates[2], baseline_rates[1], places=8)

    def test_backbone_tokens_and_position_preservation(self):
        torch.manual_seed(3)
        model = TransReID(img_size=(32, 32), patch_size=16, stride_size=16,
                          embed_dim=48, depth=12, num_heads=4, num_classes=0,
                          drop_path_rate=0., specific_tokens=True)
        self.assertEqual(tuple(model.pos_embed.shape), (1, 5, 48))
        self.assertEqual(tuple(model.specific_pos_embed.shape), (1, 1, 48))
        self.assertTrue(torch.equal(model.specific_tokens[0], model.specific_tokens[1]))
        images = torch.randn(3, 3, 32, 32)
        shared, specific, patch11 = model(images, modal_label=torch.tensor([0, 1, 2]))
        self.assertEqual(tuple(shared.shape), (3, 48))
        self.assertEqual(tuple(specific.shape), (3, 48))
        self.assertEqual(tuple(patch11.shape), (3, 4, 48))

    def test_old_backbone_checkpoint_maps_without_position_shift(self):
        options = dict(img_size=(32, 32), patch_size=16, stride_size=16,
                       embed_dim=48, depth=12, num_heads=4, num_classes=0,
                       drop_path_rate=0.)
        old = TransReID(**options)
        adapted = TransReID(**options, specific_tokens=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old.pth'
            torch.save(old.state_dict(), path)
            adapted.load_param(str(path))
        self.assertTrue(torch.equal(old.pos_embed, adapted.pos_embed))
        self.assertTrue(torch.equal(old.cls_token, adapted.cls_token))
        self.assertTrue(torch.equal(old.blocks[-1].attn.qkv.weight,
                                    adapted.blocks[-1].attn.qkv.weight))
        self.assertTrue(torch.count_nonzero(adapted.specific_pos_embed) == 0)

    def test_full_size_graph_contract(self):
        module = DLFM(dim=768, heads=12, graph_k=40)
        shared = torch.randn(2, 768)
        specific = torch.randn(2, 768)
        patches = torch.randn(2, 128, 768)
        queries_s = torch.randn(2, 16, 768)
        queries_p = torch.randn(2, 16, 768)
        with torch.no_grad():
            result = module(shared, specific, patches, queries_s, queries_p)
        self.assertEqual(tuple(result.shape), (2, 768))
        self.assertTrue(torch.isfinite(result).all())

    def test_graph_gradients_and_fusion(self):
        torch.manual_seed(4)
        dim, batch, queries = 48, 6, 4
        shared = torch.randn(batch, dim, requires_grad=True)
        specific = torch.randn(batch, dim, requires_grad=True)
        patches = torch.randn(batch, 12, dim, requires_grad=True)
        modalities = torch.tensor([0, 1, 2, 0, 1, 2])
        etgm = ETGM(dim=dim, experts=3, top_k=2, queries=queries, heads=4)
        dlfm = DLFM(dim=dim, heads=4, graph_k=5)
        q_shared, q_specific, balance = etgm(shared, specific, modalities)
        self.assertEqual(tuple(q_shared.shape), (batch, queries, dim))
        self.assertEqual(tuple(q_specific.shape), (batch, queries, dim))
        local = dlfm(shared, specific, patches, q_shared, q_specific)
        self.assertEqual(tuple(local.shape), (batch, dim))
        fusion = make_retrieval_feature(shared, local, detach_global_for_loss=True)
        self.assertEqual(tuple(fusion.shape), (batch, 2 * dim))
        self.assertTrue(torch.allclose(fusion,
                          make_retrieval_feature(shared, local), atol=1e-6))
        self.assertTrue(torch.allclose(fusion.norm(dim=1), torch.ones(batch), atol=1e-5))
        (fusion.mul(torch.randn_like(fusion)).sum() + balance).backward()
        self.assertIsNone(shared.grad)
        self.assertIsNone(patches.grad)
        self.assertIsNotNone(specific.grad)
        self.assertGreater(specific.grad.abs().sum().item(), 0)
        self.assertGreater(dlfm.shared_gcn.weight.grad.abs().sum().item(), 0)
        self.assertGreater(dlfm.specific_gcn.weight.grad.abs().sum().item(), 0)
        self.assertGreater(etgm.shared.gate.weight.grad.abs().sum().item(), 0)
        self.assertGreater(etgm.ir.gate.weight.grad.abs().sum().item(), 0)

    def test_full_retrieval_is_independent_of_eval_batch_size(self):
        torch.manual_seed(9)
        shared = torch.randn(6, 48)
        specific = torch.randn(6, 48)
        patches = torch.randn(6, 12, 48)
        modalities = torch.tensor([0, 1, 2, 0, 1, 2])
        etgm = ETGM(dim=48, experts=3, top_k=2, queries=4, heads=4).eval()
        dlfm = DLFM(dim=48, heads=4, graph_k=5).eval()

        def extract(start, stop):
            selected = slice(start, stop)
            qs, qp, _ = etgm(shared[selected], specific[selected], modalities[selected])
            local = dlfm(shared[selected], specific[selected], patches[selected], qs, qp)
            return make_retrieval_feature(shared[selected], local)

        with torch.no_grad():
            whole = extract(0, 6)
            chunks = torch.cat((extract(0, 2), extract(2, 4), extract(4, 6)), dim=0)
        self.assertTrue(torch.allclose(whole, chunks, atol=1e-5, rtol=1e-5))


if __name__ == '__main__':
    unittest.main()
