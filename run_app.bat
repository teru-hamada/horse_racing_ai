@echo off
rem セットアップ済みの仮想環境で Horse Racing AI の Streamlit アプリを起動する。
rem 初回起動前に、同じフォルダーの setup_windows.bat を実行しておく。

rem 環境変数の変更をバッチ内に限定し、末尾の endlocal で元に戻す。
setlocal
rem バッチ自身のフォルダーへ移動し、.venv と app.py の参照先をそろえる。
rem %~dp0 はバッチのドライブとフォルダー。引用符は空白を含むパスへの対策。
rem 元の作業フォルダーは pushd で保存し、終了時の popd で復元する。
pushd "%~dp0"

rem 仮想環境の Python がなければ、セットアップを案内して起動を中止する。
rem ここでは実行ファイルの存在のみ確認し、パッケージの導入状況は検証しない。
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Virtual environment was not found.
  echo Run setup_windows.bat first.
  rem ダブルクリック起動でもエラーを読めるようにキー入力を待つ。
  pause
  popd
  rem このバッチを終了し、呼び出し元へ異常終了コード 1 を返す。
  exit /b 1
)

rem 仮想環境の Python から Streamlit を実行し、app.py を Web アプリとして起動する。
rem activate は不要。Python を直接指定して、セットアップ時と同じ環境を使用する。
rem サーバーが終了するまでこのコマンドで待機する。接続 URL は実行時の表示を確認する。
echo Starting Horse Racing AI...
".venv\Scripts\python.exe" -m streamlit run app.py

rem 終了コードが 1 以上の場合はエラーを表示し、確認のためキー入力を待つ。
rem この分岐では終了コードを明示的に設定せず、表示と確認待ちを行う。
if errorlevel 1 (
  echo.
  echo ERROR: The application stopped with an error.
  pause
)

rem 起動前の作業フォルダーと環境変数のスコープを復元する。
popd
endlocal
