import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class TestWhisperTranscriber(unittest.TestCase):
    """Whisper文字起こしシステムのテスト"""

    def setUp(self):
        """各テストの前に実行される準備処理"""
        # テスト用の一時フォルダ
        self.test_dir = Path("test_temp")
        self.input_folder = self.test_dir / "input"
        self.output_folder = self.test_dir / "output"

        # テスト前にクリーンアップ
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def tearDown(self):
        """各テストの後に実行されるクリーンアップ処理"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_フォルダが自動作成される(self):
        """入力フォルダと出力フォルダが自動作成されることを確認"""
        # この時点ではフォルダは存在しない
        self.assertFalse(self.input_folder.exists())
        self.assertFalse(self.output_folder.exists())

        # whisper_transcriber モジュールをインポートして実行
        from whisper_transcriber import setup_folders
        setup_folders(str(self.input_folder), str(self.output_folder))

        # フォルダが作成されていることを確認
        self.assertTrue(self.input_folder.exists())
        self.assertTrue(self.output_folder.exists())

    def test_対応する拡張子のファイルのみが検出される(self):
        """対応する拡張子(.mp4, .m4a, .mp3, .wav, .mov)のファイルのみが処理対象になることを確認"""
        from whisper_transcriber import get_media_files

        # テスト用フォルダを作成
        self.input_folder.mkdir(parents=True)

        # 対応する拡張子のファイルを作成
        (self.input_folder / "video1.mp4").touch()
        (self.input_folder / "audio1.m4a").touch()
        (self.input_folder / "audio2.mp3").touch()
        (self.input_folder / "audio3.wav").touch()
        (self.input_folder / "video2.mov").touch()

        # 対応しない拡張子のファイルを作成
        (self.input_folder / "document.txt").touch()
        (self.input_folder / "image.jpg").touch()
        (self.input_folder / "data.json").touch()

        # ファイルを取得
        files = get_media_files(str(self.input_folder))

        # 対応する拡張子のファイルのみが取得されることを確認
        self.assertEqual(len(files), 5)
        file_names = [os.path.basename(f) for f in files]
        self.assertIn("video1.mp4", file_names)
        self.assertIn("audio1.m4a", file_names)
        self.assertIn("audio2.mp3", file_names)
        self.assertIn("audio3.wav", file_names)
        self.assertIn("video2.mov", file_names)

        # 対応しない拡張子のファイルが含まれないことを確認
        self.assertNotIn("document.txt", file_names)
        self.assertNotIn("image.jpg", file_names)
        self.assertNotIn("data.json", file_names)

    def test_文字起こし結果がタイムスタンプ付きで出力される(self):
        """文字起こし結果がタイムスタンプ付きでテキストファイルに出力されることを確認"""
        from whisper_transcriber import transcribe_file

        # テスト用フォルダを作成
        self.input_folder.mkdir(parents=True)
        self.output_folder.mkdir(parents=True)

        # テスト用の音声ファイル（空でOK）
        test_audio = self.input_folder / "test.mp3"
        test_audio.touch()

        # モックセグメントを作成
        mock_segment1 = Mock()
        mock_segment1.start = 0.0
        mock_segment1.end = 2.5
        mock_segment1.text = "こんにちは"

        mock_segment2 = Mock()
        mock_segment2.start = 2.5
        mock_segment2.end = 5.0
        mock_segment2.text = "世界"

        # モックモデルを作成
        mock_model = Mock()
        mock_info = Mock()
        mock_model.transcribe.return_value = ([mock_segment1, mock_segment2], mock_info)

        # 文字起こし実行
        transcribe_file(mock_model, str(test_audio), str(self.output_folder))

        # 出力ファイルが作成されたことを確認
        output_file = self.output_folder / "test.txt"
        self.assertTrue(output_file.exists())

        # 出力内容を確認
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()

        # タイムスタンプとテキストが正しく出力されていることを確認
        self.assertIn("[0.00s -> 2.50s] こんにちは", content)
        self.assertIn("[2.50s -> 5.00s] 世界", content)


if __name__ == "__main__":
    unittest.main()
