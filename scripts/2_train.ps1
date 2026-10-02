# Passo 2 - Training del gaussian splat con nerfstudio/gsplat.
# Uso:  .\scripts\2_train.ps1 -Scene arco            (splatfacto, 30k iterazioni)
#       .\scripts\2_train.ps1 -Scene arco -Method splatfacto-big
# Argomenti in piu' vengono passati a ns-train, es.  --max-num-iterations 7000
# Durante il training il viewer e' su http://localhost:7007
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Scene = 'arco',
    [ValidateSet('splatfacto', 'splatfacto-big', 'splatfacto-mcmc')][string]$Method = 'splatfacto',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Extra
)
. "$PSScriptRoot\..\activate.ps1"
ns-train $Method --data "$PSScriptRoot\..\data\$Scene" --output-dir "$PSScriptRoot\..\outputs" `
    --experiment-name $Scene --viewer.quit-on-train-completion True @Extra
if ($LASTEXITCODE -ne 0) { throw "ns-train fallito (exit $LASTEXITCODE)" }
