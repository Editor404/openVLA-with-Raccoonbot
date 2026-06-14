import importlib.util
import unittest
from pathlib import Path


MUJOCO_DIR = Path(__file__).resolve().parent
MODULE_PATH = MUJOCO_DIR / "raccoon_grasp_multicolor_scene_dataset.py"
SPEC = importlib.util.spec_from_file_location("raccoon_grasp_dataset", MODULE_PATH)
DATASET_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATASET_MODULE)
SyncSimRaccoonDataset = DATASET_MODULE.SyncSimRaccoonDataset


class RendererCloseCompatTest(unittest.TestCase):
    def test_renderer_without_close_method_is_supported(self):
        dataset = SyncSimRaccoonDataset.__new__(SyncSimRaccoonDataset)
        dataset.viewer = None
        dataset.renderer = object()

        dataset.close()

        self.assertIsNone(dataset.renderer)

    def test_renderer_close_method_is_called_when_available(self):
        class Renderer:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        dataset = SyncSimRaccoonDataset.__new__(SyncSimRaccoonDataset)
        dataset.viewer = None
        renderer = Renderer()
        dataset.renderer = renderer

        dataset.close()

        self.assertTrue(renderer.closed)
        self.assertIsNone(dataset.renderer)


if __name__ == "__main__":
    unittest.main()
