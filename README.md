# 無料Kaggle batch launcher

このlauncherは[公開GitHub repository](https://github.com/zuka-0123/recipe-kaggle-launcher)で管理します。Repositoryには汎用コードだけを保存し、レシピ、画像、Canonical JSON、private URLは入れません。標準の`ubuntu-latest` runnerでKaggle Notebookを起動します。GitHub ActionsではAI推論を行いません。

GitHub Secretsには`CF_WORKER_URL`、`CF_LAUNCHER_TOKEN`、`KAGGLE_KERNEL_OWNER`、`KAGGLE_KERNEL_SLUG`とKaggle認証を設定します。Kaggle認証は`KAGGLE_API_TOKEN`、または`KAGGLE_USERNAME`と`KAGGLE_KEY`を使います。KaggleのGPU利用条件、インターネット利用、API認証を確認してください。指定slugがHTTP 404 Not Foundを返した場合は、初回pushでprivate Notebookを作成します。認証エラーや判別できない状態では停止します。日常利用でNotebookを開く必要はありません。

Cloudflareが`repository_dispatch`の`recipe_batch`を送ります。Payloadは`batch_id`だけです。Launcherは公開repositoryであることと、固定Notebookが実行中でないことを確認してbatchをclaimします。生成するNotebookはprivate、GPUは`NvidiaTeslaT4`、上限は110分です。短期batch tokenだけを一時Notebookへ渡し、GitHub runnerの一時ファイルを起動後に削除します。Notebookのprivate versionには有効期限後のtokenが残り得ます。長期Worker権限やGoogle Drive認証は渡しません。

NotebookはCloudflareから対象batch、Schema、抽出ルールを取得します。LLMの初期値は`Qwen/Qwen2.5-7B-Instruct`のNF4 4bit量子化で、計算精度はfp16です。`llm_load_in_4bit`はbooleanで、既定値はtrueです。量子化に失敗した場合はエラーを返し、無量子化へ自動で切り替えません。VLMは`Qwen/Qwen2.5-VL-3B-Instruct`のfp16、ASRは`faster-whisper`の`small`を使います。Modelは必要時に切り替え、同時に複数のmodelをGPUへ置きません。画像入力では完全なSchemaを、文字入力では必要な形とenumを抽出指示へ渡します。原典、画像、model cacheは`/tmp`へ置き、終了時に削除します。結果だけをCloudflareへ送って終了します。[Transformers 4.57.1の量子化仕様](https://huggingface.co/docs/transformers/v4.57.1/quantization/bitsandbytes)に従い、`bitsandbytes==0.48.1`を使います。

YouTubeは概要欄と字幕から先に候補を作ります。料理名、材料、手順が不足する場合やJSONを生成できない場合だけ音声を取得します。フレームはユーザーが秒数を指定し、候補に不足がある場合だけ最大6枚を取得します。WebはRecipe構造化データを優先して、本文では広告や関連記事を除きます。画像はbatch専用API経由でGoogle Driveから取得します。画像は15MBまで、本文は設定上限120,000文字まで、動画音声は設定上限2,700秒まで受け付けます。LLM/VLMの入力が24,000 tokenを超える場合は、原典を切り捨てずエラーを返します。推論では外部の有料AI APIを呼びません。

無料T4が使えない場合はretry可能なエラーを返します。起動失敗や実行中のNotebookもCloudflareへ通知し、Cloudflareの再試行回数・待ち時間制限に従います。有料GPU、他のaccelerator、Colab、自動課金サービスへのfallbackはありません。無料quota不足を解消する保証はありません。

`python -m unittest discover -s tests -v`で48件すべて合格しました。テストではmodelの取得と推論をmockしています。従来の3B構成では実際のKaggle T4 batchを動作確認に使いましたが、7Bの4bit構成は実GPU検証前です。T4割り当て、modelの初回取得時間、抽出品質、YouTubeの字幕・音声取得も継続して確認します。NotebookへのSecrets登録はKaggle CLIで対応していないため、今回の短期tokenはprivate sourceへ注入します。[Kaggle公式CLI仕様](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)と[metadata仕様](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)に合わせています。

Colabで手動デバッグするときは同じworker codeを一時環境に置きます。自動起動や常駐機能はありません。


