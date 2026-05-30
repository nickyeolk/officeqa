Set-Location $PSScriptRoot

$claude    = "anthropic/claude-haiku-4-5"
$openaiM   = "openai/gpt-5.4-mini"
$googleM   = "google/gemini-3.5-flash"
$microsoftM = "openai/gpt-5.4-mini"

python run_eval.py --sdk claude    --model $claude     --subset pro
python run_eval.py --sdk openai    --model $openaiM    --subset pro
python run_eval.py --sdk google    --model $googleM    --subset pro
python run_eval.py --sdk microsoft --model $microsoftM --subset pro

python score_results.py
