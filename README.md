# 無料Kaggle batch launcher

このlauncherは[公開GitHub repository](https://github.com/zuka-0123/recipe-kaggle-launcher)で管理します。Repositoryには汎用コードだけを保存し、レシピ、画像、Canonical JSON、private URLは入れません。標準の`ubuntu-latest` runnerでKaggle Notebookを起動します。GitHub ActionsではAI推論を行いません。

GitHub Secretsには`CF_WORKER_URL`、`CF_LAUNCHER_TOKEN`、`KAGGLE_KERNEL_OWNER`、`KAGGLE_KERNEL_SLUG`とKaggle認証を設定します。Kaggle認証は`KAGGLE_API_TOKEN`、または`KAGGLE_USERNAME`と`KAGGLE_KEY`を使います。KaggleのGPU利用条件、インターネット利用、API認証を確認してください。指定slugがHTTP 404 Not Foundを返した場合は、初回pushでprivate Notebookを作成します。認証エラーや判別できない状態では停止します。日常利用でNotebookを開く必要はありません。

Cloudflareが`repository_dispatch`の`recipe_batch`を送ります。Payloadは`batch_id`だけです。Launcherは公開repositoryであることと、固定Notebookが実行中でないことを確認してbatchをclaimします。生成するNotebookはprivate、GPUは`NvidiaTeslaT4`、上限は110分です。短期batch tokenだけを一時Notebookへ渡し、GitHub runnerの一時ファイルを起動後に削除します。Notebookのprivate versionには有効期限後のtokenが残り得ます。長期Worker権限やGoogle Drive認証は渡しません。

NotebookはCloudflareから対象batch、Schema、抽出ルールを取得します。文章LLMの初期値は`Qwen/Qwen3-14B-AWQ`のAWQ 4bitです。CUDA上でfp16計算を行い、CPUへ全面offloadしません。Thinkingはtokenizerの`enable_thinking=False`で無効にします。本番のCloudflareモデル設定も14Bへ変更しました。native AWQ CUDAカーネルとefficient SDPAを、無料T4で動作確認した設定で使います。14Bの読込時だけ、必要なら既存PyTorchに対応したカーネルを最大420秒でビルドします。旧7Bは`llm_model=Qwen/Qwen2.5-7B-Instruct`と`llm_load_in_4bit=true`だけでNF4へ戻せます。AWQをbitsandbytesで再量子化せず、自動の別モデルfallbackも行いません。量子化に失敗した場合はエラーを返し、無量子化へ自動で切り替えません。VLMは`Qwen/Qwen2.5-VL-3B-Instruct`のfp16、ASRは`faster-whisper`の`small`を使います。Modelは必要時に切り替え、同時に複数のmodelをGPUへ置きません。画像入力では完全なSchemaを、文字入力では必要な形とenumを抽出指示へ渡します。原典、画像、model cacheは`/tmp`へ置き、終了時に削除します。結果だけをCloudflareへ送って終了します。[Transformers 4.57.1の量子化仕様](https://huggingface.co/docs/transformers/v4.57.1/quantization/bitsandbytes)に従い、`bitsandbytes==0.48.1`を使います。

YouTubeは概要欄と字幕から先に候補を作ります。料理名、材料、手順が不足する場合やJSONを生成できない場合だけ音声を取得します。ただし、生成時間・出力token上限による途中切れでは音声を追加取得しません。フレームはユーザーが秒数を指定し、候補に不足がある場合だけ最大6枚を取得します。WebはRecipe構造化データを優先して、本文では広告や関連記事を除きます。画像はbatch専用API経由でGoogle Driveから取得します。画像は15MBまで、本文は設定上限120,000文字まで、動画音声は設定上限2,700秒まで受け付けます。14B AWQは入力12,000 token、旧7BとVLMは24,000 tokenを超える場合にエラーを返します。本文上限とは別に検査します。model入力が上限を超える場合は、原典を切り捨てずエラーを返します。推論では外部の有料AI APIを呼びません。

無料T4が使えない場合はretry可能なエラーを返します。起動失敗や実行中のNotebookもCloudflareへ通知し、Cloudflareの再試行回数・待ち時間制限に従います。有料GPU、他のaccelerator、Colab、自動課金サービスへのfallbackはありません。無料quota不足を解消する保証はありません。

LLM生成中はT4でefficient SDPAだけを使います。固定したTransformersのK/V展開経路を一時的に使い、GQAが大きなメモリを使うmath方式へ進むのを防ぎます。処理後は設定を戻します。対応しない環境では`model_api_incompatible`として停止します。原典の切り捨てや別GPUへの切り替えはありません。

launcherのCPUテスト82件が合格しました。modelの取得と推論はmockしています。実際の無料T4では14BのロードとJSON生成を確認しました。MVPでは代表例の主要フローを確認し、性能ベンチマークやモデル選定の追加検証は行いません。原文にない人数・時間・火加減や字幕由来の分量誤り、根拠参照の不足が候補に残る場合があるため、人が確認・修正してから正式保存します。取得した原文と一致しない引用はnullにし、存在しない根拠IDは受け入れません。

ASRはPyAV 18.1.0に固定しています。AWQは公式と同じ旧クラス名aliasでAutoAWQを読み込みます。自動起動の日次上限は変更していません。NotebookへのSecrets登録はKaggle CLIで対応していないため、短期tokenはprivate sourceへ注入します。[Kaggle公式CLI仕様](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)と[metadata仕様](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)に合わせています。

14Bのgenerationは[公式の非Thinking設定](https://huggingface.co/Qwen/Qwen3-14B-AWQ)であるtemperature 0.7 / top_p 0.8 / top_k 20 / min_p 0をseed 42で使います。出力上限は6,144 token、生成上限は900秒（4,000 token以上のYouTubeは1,200秒）で、batchの残り時間も守ります。14Bのbatch上限は105分で、Notebookの110分上限とtokenの120分期限を守ります。生成されたthinkingタグは削って受け入れず、raw outputを保持したエラーとして確認対象にします。

Colabで手動デバッグするときは同じworker codeを一時環境に置きます。自動起動や常駐機能はありません。
