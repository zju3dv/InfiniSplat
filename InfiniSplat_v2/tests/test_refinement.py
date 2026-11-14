"""CPU compatibility, membership, chunking and weight-contract regression checks."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from safetensors.torch import save_file
from infinisplat.gaussians import merge
from infinisplat.pipeline import Pipeline, sha256
from infinisplat.refinement.loader import CONTRACT, load_refiner, validate_config
from infinisplat.refinement.knn import knn_dependency_chunks
from infinisplat.refinement.refiner import RefinerBundle

BASE = dict(backend='da3', format_version=2, weights_sha256=CONTRACT['base_weights_sha256'],
            **{k:CONTRACT[k] for k in ('normalization','merge','voxel_multiplier','opacity_threshold')}, eps2d=.01)


class RefinementTests(unittest.TestCase):
    def test_import_does_not_load_optional_dependencies(self):
        script = "import infinisplat.pipeline,sys; assert not any(k.split('.')[0] in " + repr(
            {'gsplat','spconv','lpips','lightning','hydra'}) + " for k in sys.modules); assert 'infinisplat.refinement' not in sys.modules"
        subprocess.run([sys.executable, '-c', script], check=True)

    def test_membership_preserves_merge_and_feature_order(self):
        means = torch.tensor([[0.,0,1],[.01,0,1],[1.,0,1],[2.,0,1],[3.,0,1]])
        scene = dict(means=means, scales=torch.full((5,3),.1),
            quats=torch.tensor([[1.,0,0,0]]).repeat(5,1), opacities=torch.tensor([.4,.8,.5,.001,.6]),
            sh=torch.zeros(5,1,3),base_rgb=torch.zeros(5,3))
        scene['means'][-1,0] = float('nan')
        plain, stats = merge(scene)
        tapped, stats2, m = merge(scene, return_membership=True)
        self.assertEqual(stats, stats2)
        for key in plain: self.assertTrue(torch.equal(plain[key], tapped[key]))
        self.assertEqual(m['keep'].tolist(), [True,True,True,True,False])
        self.assertEqual(m['counts'].tolist(), [2.,1.,1.])
        self.assertEqual(m['visible'].tolist(), [True,True,False])
        features = torch.arange(5.).reshape(-1,1).repeat(1,8)
        merged = torch.zeros(3,8).index_add_(0,m['inverse'],features[m['keep']])
        actual = (merged/m['counts'][:,None])[m['visible']]
        torch.testing.assert_close(actual, torch.tensor([[.5]*8,[2.]*8]), rtol=0, atol=0)

    def test_chunk_closure_matches_full_attention(self):
        torch.manual_seed(14)
        bundle=RefinerBundle().eval().requires_grad_(False)
        n=25
        grid=torch.stack([torch.arange(n),torch.zeros(n,dtype=torch.long),torch.zeros(n,dtype=torch.long)],-1)
        ids=torch.arange(n)
        graph=torch.stack([ids,(ids-1).clamp_min(0),(ids+1).clamp_max(n-1)],-1)
        features=torch.randn(n,59)
        chunks=knn_dependency_chunks(grid,graph,blocks=4,interior_limit=5,total_limit=20)
        with torch.no_grad():
            actual=bundle.refiner.residual([features],graph,chunks)
            expected=bundle.refiner.network(features,graph)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
        self.assertGreater(len(chunks),1)

    def test_strict_loader_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            bundle=RefinerBundle()
            save_file(bundle.state_dict(), str(path/'refiner.safetensors'))
            cfg=dict(CONTRACT,source_version='fixed_center_knn_v1',weights_sha256=sha256(path/'refiner.safetensors'))
            (path/'config.json').write_text(json.dumps(cfg))
            with patch('infinisplat.refinement.feedback.require_renderer'):
                loaded,_=load_refiner(path,BASE,'cpu')
            self.assertFalse(loaded.training)
            self.assertTrue(all(not v.requires_grad for v in loaded.parameters()))
            self.assertEqual(len(loaded.state_dict()),56)
            for key,value in bundle.state_dict().items():self.assertTrue(torch.equal(value,loaded.state_dict()[key]))
            with self.assertRaisesRegex(ValueError,'exact released'):
                validate_config(cfg,dict(BASE,weights_sha256='0'*64))
            with self.assertRaisesRegex(ValueError,'contract'):
                validate_config(dict(cfg,knn_blocks=5),BASE)
            bad=bundle.state_dict();bad['projector.weight']=bad['projector.weight'][:1]
            save_file(bad,str(path/'refiner.safetensors'))
            cfg['weights_sha256']=sha256(path/'refiner.safetensors')
            (path/'config.json').write_text(json.dumps(cfg))
            with patch('infinisplat.refinement.feedback.require_renderer'), self.assertRaisesRegex(ValueError,'shapes'):
                load_refiner(path,BASE,'cpu')
            with (path/'refiner.safetensors').open('ab') as stream:stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'integrity'):
                load_refiner(path,BASE,'cpu')

    def test_rejects_malformed_manifest_types_before_reading_weights(self):
        cfg=dict(CONTRACT, source_version='fixed_center_knn_v1', weights_sha256='0'*64)
        invalid=[None, 17, list(cfg), dict(cfg,format_version=True),
                 dict(cfg,knn_blocks=4.0), dict(cfg,default_steps=3.0),
                 dict(cfg,update_position=0), dict(cfg,update_opacity=1),
                 dict(cfg,rasterizer=dict(cfg['rasterizer'],packed=1))]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            for value in invalid:
                with self.subTest(config=value):
                    (path/'config.json').write_text(json.dumps(value))
                    with self.assertRaises(ValueError):
                        load_refiner(path,BASE,'cpu')

    def test_missing_renderer_and_invalid_tensor_file_report_clear_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            save_file(RefinerBundle().state_dict(),str(path/'refiner.safetensors'))
            cfg=dict(CONTRACT,source_version='fixed_center_knn_v1',weights_sha256=sha256(path/'refiner.safetensors'))
            (path/'config.json').write_text(json.dumps(cfg))
            with patch.dict(sys.modules,{'gsplat':None}), self.assertRaisesRegex(RuntimeError,'optional renderer'):
                load_refiner(path,BASE,'cpu')
            (path/'refiner.safetensors').write_bytes(b'not a safetensors file')
            cfg['weights_sha256']=sha256(path/'refiner.safetensors')
            (path/'config.json').write_text(json.dumps(cfg))
            with patch('infinisplat.refinement.feedback.require_renderer'), self.assertRaisesRegex(ValueError,'tensor file'):
                load_refiner(path,BASE,'cpu')

    def test_cli_reports_invalid_refiner_without_traceback(self):
        from infinisplat.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/'config.json').write_text('null')
            pipeline=Pipeline(torch.nn.Identity(),BASE,'cpu',refiner=str(path))
            argv=['infinisplat','--model','unused','--refiner',str(path),
                  '--images','a.png','b.png','--output',str(path/'result'),'--no-html']
            stderr=io.StringIO()
            with patch.object(sys,'argv',argv), patch('infinisplat.cli.Pipeline.from_pretrained',return_value=pipeline):
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                    main()
            self.assertEqual(error.exception.code,2)
            self.assertIn('Unsupported refiner architecture',stderr.getvalue())
            self.assertNotIn('Traceback',stderr.getvalue())
            self.assertFalse((path/'result').exists())

    def test_recurrent_state_resets_when_feedback_is_fixed(self):
        torch.manual_seed(23)
        bundle=RefinerBundle().eval().requires_grad_(False)
        n=17
        means=torch.randn(n,3)*.1;means[:,2]+=2
        scene=dict(means=means,scales=torch.full((n,3),.02),opacities=torch.full((n,),.5),
                   quats=torch.tensor([[1.,0,0,0]]).repeat(n,1),sh=torch.zeros(n,1,3))
        initial={k:v.clone() for k,v in scene.items()}
        features=torch.randn(n,8)
        feedback=[(torch.zeros(n,c),torch.zeros(n,c)) for c in (3,1,3,3)]
        args=(scene,features,torch.zeros(2,3,14,28),torch.eye(4).repeat(2,1,1),
              torch.eye(3).repeat(2,1,1),.05)
        with patch('infinisplat.refinement.refiner.Feedback.__call__',return_value=feedback):
            first=bundle.refiner(*args,steps=3)
            second=bundle.refiner(*args,steps=3)
        for key in scene:
            self.assertTrue(torch.equal(first[key],second[key]),key)
            self.assertTrue(torch.equal(scene[key],initial[key]),key)
        self.assertTrue(all(p.grad is None for p in bundle.parameters()))

    def test_pipeline_modes_zero_steps_and_lazy_loading(self):
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.calls=[]
            def predict_cameras(self,pixels):return torch.eye(4).repeat(1,2,1,1),K[None]
            def forward(self,pixels,poses,intrinsics,**kwargs):
                self.calls.append((poses,kwargs))
                return dict(c2w=torch.eye(4).repeat(2,1,1),intrinsics=K,stats={})
        K=torch.tensor([[[100.,0,55.5],[0,100.,55.5],[0,0,1.]]]).repeat(2,1,1)
        with tempfile.TemporaryDirectory() as tmp:
            image=Path(tmp)/'image.png';Image.new('RGB',(112,112)).save(image)
            model=Model();pipeline=Pipeline(model,BASE,'cpu',refiner='/missing-refiner')
            cameras=dict(c2w=np.repeat(np.eye(4,dtype=np.float32)[None],2,axis=0),intrinsics=K.numpy())
            for mode in ('provided','da3-pose','pose-free'):
                result=pipeline.reconstruct([image]*2,long_side=112,pose_mode=mode,
                    cameras=cameras if mode=='provided' else None,refinement_steps=0)
                self.assertEqual(result['metadata']['pose_mode'],mode)
                self.assertNotIn('refinement',result['metadata'])
                self.assertEqual(model.calls[-1][1],{})
            self.assertIsNone(pipeline._refiner)
            with self.assertRaisesRegex(ValueError,'panorama'):
                pipeline.reconstruct([image]*2,cameras=cameras,panorama_shared_center=True)
            for invalid in (-1,4,True,1.5):
                with self.assertRaises(ValueError):pipeline.reconstruct([image]*2,refinement_steps=invalid)
            pipeline.refiner_source=None
            with self.assertRaisesRegex(ValueError,'requires'):
                pipeline.reconstruct([image]*2,refinement_steps=1)


if __name__ == '__main__':unittest.main()
