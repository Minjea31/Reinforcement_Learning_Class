import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from web.mission_replays import prepare,publish,snapshot,checkpoint_source


class MissionReplayTests(unittest.TestCase):
    def test_inherited_stage_one_uses_the_frozen_pickup_source(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            torch.save(dict(steps=20_000_000,env_config=dict(stage=1)),run/'pickup_source.pt')
            self.assertEqual(checkpoint_source(run,1).name,'pickup_source.pt')
            self.assertEqual(snapshot(run,1)[1]['steps'],20_000_000)
            self.assertIsNone(snapshot(run,2))

    def test_separate_credentials_do_not_change_the_published_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=dict(stage=2,run_name='current_run',entries=[])
            class Response:
                status=200
                def __enter__(self):return self
                def __exit__(self,*args):pass
            class Client:
                def open(self,request,timeout):
                    self.body=json.loads(request.data);return Response()
            client=Client()
            with patch('web.mission_replays.read_credentials',return_value=dict(origin='https://private.example',site_token='test-token')) as read,patch('web.mission_replays.request.build_opener',return_value=client):
                publish(root,root,manifest,connection_dir='old_connection')
            read.assert_called_once_with('old_connection')
            self.assertEqual(client.body['run_name'],'current_run')

    def test_best_checkpoint_is_frozen_and_stage_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            torch.save(dict(steps=12,env_config=dict(stage=1),policy=dict(weight=torch.ones(2))),run/'stage1_best.pt')
            torch.save(dict(steps=999,env_config=dict(stage=2)),run/'latest.pt')
            folder,checkpoint,digest=snapshot(run,1)
            self.assertEqual(checkpoint['steps'],12)
            self.assertEqual((folder/'checkpoint.pt').read_bytes(),(run/'stage1_best.pt').read_bytes())
            self.assertEqual(len(digest),64)
            self.assertIsNone(snapshot(run,3))
            torch.save(dict(env_config=dict(stage=1)),run/'stage2_best.pt')
            with self.assertRaises(ValueError):snapshot(run,2)

    def test_uploads_replays_before_manifest_and_never_uploads_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'rank-01').mkdir()
            entry=dict(rank=1,video='rank-01/replay.mp4',poster='rank-01/poster.jpg',metadata='rank-01/replay.json')
            for field in ('video','poster','metadata'):(root/entry[field]).write_bytes(b'asset')
            (root/'checkpoint.pt').write_bytes(b'private model')
            manifest=dict(stage=2,asset_base='/api/mission-assets/stage-2/'+'a'*64+'/',entries=[entry])
            requests=[]
            class Response:
                status=200
                def __enter__(self):return self
                def __exit__(self,*args):pass
            class Client:
                def open(self,request,timeout):requests.append(request);return Response()
            with patch('web.mission_replays.read_credentials',return_value=dict(origin='https://private.example',site_token='test-token')),patch('web.mission_replays.request.build_opener',return_value=Client()):
                publish(root,root,manifest)
            self.assertEqual([r.method for r in requests],['PUT','PUT','PUT','POST'])
            self.assertTrue(all('checkpoint.pt' not in r.full_url for r in requests))
            self.assertEqual(requests[-1].full_url,'https://private.example/api/missions/2')
            self.assertEqual(json.loads(requests[-1].data)['stage'],2)


if __name__=='__main__':unittest.main()
