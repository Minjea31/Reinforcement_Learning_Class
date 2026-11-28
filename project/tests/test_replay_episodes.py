import json
from pathlib import Path
import tempfile
import unittest

from evaluation.export_replay import select_scenarios
from web.mission_replays import describe


class ReplayEpisodeTests(unittest.TestCase):
    def test_episode_number_is_original_evaluation_order_not_reward_rank(self):
        records=[dict(seed=100,episode_return=-2),dict(seed=101,episode_return=5),dict(seed=102,episode_return=1)]
        selected=select_scenarios(dict(records=records),None,2)
        self.assertEqual([r['episode'] for r in selected],[2,3])
        self.assertEqual([r['seed'] for r in selected],[101,102])
        self.assertNotIn('episode',records[0])

    def test_single_seed_is_not_mislabeled_as_an_evaluation_episode(self):
        self.assertIsNone(select_scenarios(None,123,1)[0].get('episode'))

    def test_cached_rankings_and_video_metadata_recover_original_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);folder=run/'cached';assets=folder/'assets';assets.mkdir(parents=True)
            (assets/'evaluation.json').write_text(json.dumps(dict(records=[dict(seed=91),dict(seed=92),dict(seed=93)])))
            entry=dict(rank=1,seed=93,metadata='rank-01/replay.json')
            (assets/'rank-01').mkdir();(assets/entry['metadata']).write_text(json.dumps(dict(seed=93)))
            (assets/'replays.json').write_text(json.dumps(dict(entries=[entry])))
            checkpoint=dict(config=dict(num_envs=1),curriculum=dict(level=0))
            _,manifest=describe(run,1,folder,checkpoint,'a'*64)
            self.assertEqual(manifest['entries'][0]['episode'],3)
            self.assertEqual(json.loads((assets/entry['metadata']).read_text())['episode'],3)


if __name__=='__main__':unittest.main()
