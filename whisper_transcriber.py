"""
Whisper文字起こしシステム
faster-whisperを使用して動画ファイルから文字起こしを行う
"""
import os
from pathlib import Path
from typing import List


# 対応するメディアファイルの拡張子
SUPPORTED_EXTENSIONS = ('.mp4', '.m4a', '.mp3', '.wav', '.mov')


def setup_folders(input_folder: str, output_folder: str) -> None:
    """
    入力フォルダと出力フォルダを作成する

    Args:
        input_folder: 入力フォルダのパス
        output_folder: 出力フォルダのパス
    """
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)


def get_media_files(input_folder: str) -> List[str]:
    """
    入力フォルダ内のメディアファイルを取得する

    Args:
        input_folder: 入力フォルダのパス

    Returns:
        メディアファイルのパスのリスト
    """
    files = []
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(SUPPORTED_EXTENSIONS):
            file_path = os.path.join(input_folder, file_name)
            files.append(file_path)
    return files


def transcribe_file(model, file_path: str, output_folder: str) -> None:
    """
    音声ファイルを文字起こしして結果をテキストファイルに出力する

    Args:
        model: Whisperモデル
        file_path: 音声ファイルのパス
        output_folder: 出力フォルダのパス
    """
    # ファイル名を取得（拡張子なし）
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]

    print(f"Processing: {file_name} ...")

    # 文字起こし実行
    segments, info = model.transcribe(file_path, beam_size=5, language="ja")

    # 出力ファイルのパス
    output_path = os.path.join(output_folder, f"{base_name}.txt")

    # 結果を書き出し
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
            print(line)  # コンソールにも表示
            f.write(line + "\n")

    print(f"Done! Output: {output_path}")
    print("-" * 30)


def main(
    input_folder: str = "input",
    output_folder: str = "output",
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16"
) -> None:
    """
    メイン処理

    Args:
        input_folder: 入力フォルダのパス
        output_folder: 出力フォルダのパス
        model_size: モデルサイズ (tiny, base, small, medium, large-v3)
        device: デバイス (cuda, cpu)
        compute_type: 計算タイプ (float16, int8)
    """
    # フォルダ作成
    setup_folders(input_folder, output_folder)

    # メディアファイルの取得
    files = get_media_files(input_folder)

    if not files:
        print(f"'{input_folder}' フォルダに動画/音声ファイルがありません。")
        return

    print(f"Running on {device} with {model_size} model.")
    print(f"Found {len(files)} file(s) to process.")
    print("-" * 30)

    # モデルのロード
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        print("Error loading model. Make sure CUDA libraries are correct.")
        print(e)
        return

    # 順次処理
    for file_path in files:
        transcribe_file(model, file_path, output_folder)

    print("All tasks finished.")


if __name__ == "__main__":
    main()
