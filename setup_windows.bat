@echo off
rem Windows 用の初回セットアップ。Python 3.12 の仮想環境に依存パッケージを導入する。
rem 事前に Python 3.12 と Python Launcher（py コマンド）をインストールしておく。
rem パッケージのダウンロードには、通常インターネット接続が必要。

rem 環境変数の変更をこのバッチ内に限定する。バッチ終了時に元の環境へ戻る。
setlocal
rem バッチ自身のフォルダーへ移動し、相対パスの基準をそろえる。
rem %~dp0 はバッチのドライブとフォルダー。引用符は空白を含むパスへの対策。
rem 元の作業フォルダーは pushd で保存し、終了時の popd で復元する。
pushd "%~dp0"

rem 手順 1: Python Launcher から Python 3.12 を実行できるか確認する。
echo [1/4] Checking Python 3.12...
rem 標準出力と標準エラー出力を破棄し、終了コードだけを確認する。
py -3.12 --version >nul 2>&1
rem errorlevel 1 は終了コードが 1 以上の場合。見つからなければ処理を中止する。
rem py -0p は Launcher が認識している Python の一覧と実行ファイルの場所を表示する。
if errorlevel 1 (
  echo ERROR: Python 3.12 was not found.
  echo Run: py -0p
  echo Then confirm that Python 3.12 is installed.
  rem ダブルクリック起動でもエラーを読めるようにキー入力を待つ。
  pause
  popd
  rem このバッチを終了し、呼び出し元へ異常終了コード 1 を返す。
  exit /b 1
)

rem 手順 2: プロジェクト専用の仮想環境 .venv を作成する。
echo [2/4] Creating virtual environment...
rem Python が既に存在する場合は再作成せず、既存の仮想環境を使用する。
rem この存在確認では、既存環境の Python バージョンや破損の有無までは検証しない。
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv ".venv"
  rem 作成に失敗した場合は末尾の共通エラー処理へ移動する。
  if errorlevel 1 goto :error
) else (
  echo Virtual environment already exists.
)

rem 手順 3: 仮想環境のパッケージ管理ツール pip を最新版へ更新する。
rem activate は使わず Python を直接指定し、導入先を .venv に固定する。
echo [3/4] Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

rem 手順 4: requirements.txt に記載された依存パッケージを同じ仮想環境に導入する。
rem 既存環境でも実行されるため、再実行時も必要な依存パッケージの導入が行われる。
echo [4/4] Installing packages...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

rem 全処理の成功を表示して確認を待ち、作業フォルダーを戻して正常終了コード 0 を返す。
echo.
echo Setup completed successfully.
echo Double-click run_app.bat to start the application.
pause
popd
exit /b 0

rem 仮想環境の作成、pip の更新、パッケージ導入に失敗した場合の共通処理。
rem 直前のエラーを確認できるよう待機し、作業フォルダーを戻して終了コード 1 を返す。
:error
echo.
echo ERROR: Setup failed. Review the messages above.
pause
popd
exit /b 1
