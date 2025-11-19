# 高速AI文字起こしシステム (Whisper)

faster-whisperを使用したGPU高速文字起こしシステムです。

## 機能

- 複数の動画・音声ファイルの一括文字起こし
- NVIDIA GPU (CUDA) による高速処理
- タイムスタンプ付きテキスト出力
- 対応フォーマット: .mp4, .m4a, .mp3, .wav, .mov

## 事前準備

### 1. FFmpegのインストール

PowerShell (管理者権限推奨) で以下を実行:

```powershell
winget install "FFmpeg (Essentials Build)"
```

インストール後、PCを再起動するか、ターミナルを再起動してパスを反映させてください。

確認コマンド:
```bash
ffmpeg -version
```

### 2. NVIDIA Driver

RTX 4060 Ti 用の最新ドライバがインストールされていることを確認してください。

## セットアップ

### 1. 仮想環境の作成

```bash
python -m venv venv
```

### 2. 仮想環境の有効化

```bash
# Windows
.\venv\Scripts\activate
```

### 3. ライブラリのインストール

```bash
pip install faster-whisper
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 使い方

### 1. 入力フォルダの準備

プロジェクトフォルダ内に `input` フォルダを作成します（初回実行時に自動作成されます）。

### 2. 動画ファイルの配置

文字起こししたい動画・音声ファイルを `input` フォルダに入れます。

### 3. 実行

```bash
python whisper_transcriber.py
```

### 4. 結果の確認

処理が完了すると `output` フォルダにテキストファイルが保存されます。

## 出力形式

```
[0.00s -> 2.50s] こんにちは
[2.50s -> 5.00s] 世界
```

各行にタイムスタンプと文字起こし結果が表示されます。

## カスタマイズ

`whisper_transcriber.py` の `main()` 関数で以下のパラメータを変更できます:

```python
main(
    input_folder="input",      # 入力フォルダ
    output_folder="output",    # 出力フォルダ
    model_size="large-v3",     # モデルサイズ
    device="cuda",             # デバイス (cuda/cpu)
    compute_type="float16"     # 計算タイプ
)
```

### モデルサイズ

- `tiny`: 最速だが精度は低い
- `base`: 高速で軽量
- `small`: バランス型
- `medium`: 高精度
- `large-v3`: 最高精度（推奨）

## テスト

```bash
python test_whisper.py
```

## トラブルシューティング

### cudnn_ops_infer64_8.dll エラー

NVIDIA cuDNN ライブラリが見つからない場合に発生します。
[NVIDIA公式サイト](https://developer.nvidia.com/cudnn)から cuDNN をダウンロードしてインストールしてください。

### zlibwapi.dll が見つからない

`zlibwapi.dll` をダウンロードして、`C:\Windows\System32` またはプロジェクトフォルダにコピーしてください。

## パフォーマンス

RTX 4060 Ti での処理速度目安:
- 1時間の動画: 約5-8分で完了

## ライセンス

MIT License
