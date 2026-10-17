<#
.SYNOPSIS
  Builds a small, clean training set for Veil (student test generator).

.DESCRIPTION
  Instead of downloading whole datasets (hundreds of MB), this pulls only the
  rows you ask for, 100 at a time, from the Hugging Face dataset viewer API,
  then converts them into Veil's instruction/input/output JSONL format.
  Several questions are grouped into one "test with an answer key" so the model
  learns the real output format.

  Sources (all small, question-focused):
    SciQ        science multiple-choice
    ARC         grade-school science multiple-choice (Easy + Challenge)
    OpenBookQA  elementary science multiple-choice
    SQuAD       short-answer questions written from a passage
    GSM8K       math word problems with step-by-step solutions (off by default)

.EXAMPLE
  ./get-veil-data.ps1
  ./get-veil-data.ps1 -Sciq 1000 -Arc 500 -OpenBook 500 -Squad 800 -Math 600
  ./get-veil-data.ps1 -Sciq 0 -Squad 0        # only ARC + OpenBookQA

  Optional: set $env:HF_TOKEN first for higher rate limits.
#>
[CmdletBinding()]
param(
    [string]$OutFile  = "data/train.jsonl",
    [int]$Sciq        = 2000,   # rows to download (0 = skip)
    [int]$Arc         = 1000,   # rows per ARC config (Easy and Challenge)
    [int]$OpenBook    = 1000,
    [int]$Squad       = 1500,
    [int]$Math        = 0,      # GSM8K rows
    [int]$GroupSize   = 4,      # questions per generated test
    [int]$Seed        = 42
)

$ErrorActionPreference = "Stop"
$rng      = [System.Random]::new($Seed)
$headers  = @{}
if ($env:HF_TOKEN) { $headers["Authorization"] = "Bearer $($env:HF_TOKEN)" }
$apiBase  = "https://datasets-server.huggingface.co/rows"
$letters  = "A", "B", "C", "D", "E", "F"
$totalDownloaded = 0

$mcqTemplates = @(
    "Create a {n}-question multiple-choice test on {subject} with an answer key.",
    "Write {n} multiple-choice questions about {subject}. Include the correct answers at the end.",
    "Generate a short {subject} quiz with {n} multiple-choice questions and an answer key."
)
$passageTemplates = @(
    "Read the passage and write {n} short-answer questions with answers.",
    "Create {n} comprehension questions from this passage, with an answer for each."
)
$mathTemplates = @(
    "Create {n} math word problems with step-by-step solutions and final answers.",
    "Write a {n}-question math test with worked solutions."
)

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Pick($arr)       { $arr[$rng.Next($arr.Count)] }
function Fill($tpl, $n, $subject = "") { $tpl.Replace("{n}", "$n").Replace("{subject}", $subject) }

# Download $Count rows of a dataset split, 100 per request, with retries.
function Get-Rows($Dataset, $Config, $Split, [int]$Count) {
    $rows   = [System.Collections.Generic.List[object]]::new()
    $offset = 0
    while ($rows.Count -lt $Count) {
        $len  = [Math]::Min(100, $Count - $rows.Count)
        $url  = "${apiBase}?dataset=$Dataset&config=$Config&split=$Split&offset=$offset&length=$len"
        $resp = $null
        for ($try = 1; $try -le 5 -and -not $resp; $try++) {
            try   { $resp = Invoke-RestMethod -Uri $url -Headers $headers -TimeoutSec 60 }
            catch { Write-Warning "Retry $try for $Dataset (offset $offset): $($_.Exception.Message)"
                    Start-Sleep -Seconds ([Math]::Pow(2, $try)) }
        }
        if (-not $resp -or -not $resp.rows -or $resp.rows.Count -eq 0) { break }
        foreach ($r in $resp.rows) { $rows.Add($r.row) }
        $offset += $resp.rows.Count
        Start-Sleep -Milliseconds 300
    }
    Write-Host "    got $($rows.Count) rows"
    $script:totalDownloaded += $rows.Count
    ,$rows
}

# Group multiple-choice items into tests with an answer key.
# Each item: @{ Q = text; Options = array; Correct = index }
function New-McqExamples($items, $subject) {
    $examples = [System.Collections.Generic.List[object]]::new()
    $items = @($items)
    for ($i = 0; $i -lt $items.Count; $i += $GroupSize) {
        $end   = [Math]::Min($i + $GroupSize, $items.Count) - 1
        $chunk = @($items[$i..$end])
        if ($chunk.Count -lt 2) { continue }
        $body = [System.Text.StringBuilder]::new()
        $key  = [System.Text.StringBuilder]::new()
        $num  = 0
        foreach ($it in $chunk) {
            $num++
            [void]$body.AppendLine("$num. $(([string]$it.Q).Trim())")
            for ($k = 0; $k -lt $it.Options.Count; $k++) {
                [void]$body.AppendLine("$($letters[$k])) $(([string]$it.Options[$k]).Trim())")
            }
            [void]$body.AppendLine()
            [void]$key.AppendLine("$num. $($letters[$it.Correct])")
        }
        $output = ($body.ToString().TrimEnd() + "`n`nAnswer key:`n" + $key.ToString().TrimEnd()).Replace("`r`n", "`n")
        $examples.Add([ordered]@{
            instruction = Fill (Pick $mcqTemplates) $chunk.Count $subject
            input       = ""
            output      = $output
        })
    }
    ,$examples
}

$all = [System.Collections.Generic.List[object]]::new()

# ---------- SciQ ----------
if ($Sciq -gt 0) {
    Write-Step "SciQ"
    $rows  = Get-Rows "allenai/sciq" "default" "train" $Sciq
    $items = foreach ($r in $rows) {
        $opts = @(@($r.correct_answer, $r.distractor1, $r.distractor2, $r.distractor3) | Sort-Object { $rng.Next() })
        $idx  = [array]::IndexOf($opts, $r.correct_answer)
        if ($idx -lt 0) { continue }
        [pscustomobject]@{ Q = $r.question; Options = $opts; Correct = $idx }
    }
    $all.AddRange((New-McqExamples $items "science"))
}

# ---------- ARC ----------
if ($Arc -gt 0) {
    foreach ($cfg in "ARC-Easy", "ARC-Challenge") {
        Write-Step $cfg
        $rows  = Get-Rows "allenai/ai2_arc" $cfg "train" $Arc
        $items = foreach ($r in $rows) {
            $texts  = @($r.choices.text)
            $labels = @($r.choices.label)
            $idx    = [array]::IndexOf($labels, [string]$r.answerKey)
            if ($idx -lt 0 -or $texts.Count -lt 2 -or $texts.Count -gt $letters.Count) { continue }
            [pscustomobject]@{ Q = $r.question; Options = $texts; Correct = $idx }
        }
        $all.AddRange((New-McqExamples $items "school science"))
    }
}

# ---------- OpenBookQA ----------
if ($OpenBook -gt 0) {
    Write-Step "OpenBookQA"
    $rows  = Get-Rows "allenai/openbookqa" "main" "train" $OpenBook
    $items = foreach ($r in $rows) {
        $texts  = @($r.choices.text)
        $labels = @($r.choices.label)
        $idx    = [array]::IndexOf($labels, [string]$r.answerKey)
        if ($idx -lt 0 -or $texts.Count -lt 2 -or $texts.Count -gt $letters.Count) { continue }
        [pscustomobject]@{ Q = $r.question_stem; Options = $texts; Correct = $idx }
    }
    $all.AddRange((New-McqExamples $items "elementary science"))
}

# ---------- SQuAD (short-answer questions from a passage) ----------
if ($Squad -gt 0) {
    Write-Step "SQuAD"
    $rows   = Get-Rows "rajpurkar/squad" "plain_text" "train" $Squad
    $groups = [ordered]@{}
    foreach ($r in $rows) {
        if (-not $groups.Contains($r.context)) { $groups[$r.context] = [System.Collections.Generic.List[object]]::new() }
        $groups[$r.context].Add($r)
    }
    foreach ($ctx in $groups.Keys) {
        $qs = @($groups[$ctx] | Select-Object -First 3)
        $sb = [System.Text.StringBuilder]::new()
        $n  = 0
        foreach ($q in $qs) {
            $answer = @($q.answers.text)[0]
            if (-not $answer) { continue }
            $n++
            [void]$sb.AppendLine("$n. $($q.question.Trim())")
            [void]$sb.AppendLine("Answer: $($answer.Trim())")
            [void]$sb.AppendLine()
        }
        if ($n -eq 0) { continue }
        $all.Add([ordered]@{
            instruction = Fill (Pick $passageTemplates) $n
            input       = $ctx.Trim()
            output      = $sb.ToString().TrimEnd().Replace("`r`n", "`n")
        })
    }
}

# ---------- GSM8K (math word problems) ----------
if ($Math -gt 0) {
    Write-Step "GSM8K"
    $rows = (Get-Rows "openai/gsm8k" "main" "train" $Math).ToArray()
    for ($i = 0; $i -lt $rows.Count; $i += 3) {
        $chunk = @($rows[$i..([Math]::Min($i + 2, $rows.Count - 1))])
        $sb = [System.Text.StringBuilder]::new()
        $n  = 0
        foreach ($r in $chunk) {
            $parts = $r.answer -split "####"
            if ($parts.Count -lt 2) { continue }
            $n++
            $solution = ($parts[0] -replace "<<[^>]*>>", "").Trim()
            [void]$sb.AppendLine("$n. $($r.question.Trim())")
            [void]$sb.AppendLine("Solution: $solution")
            [void]$sb.AppendLine("Answer: $($parts[1].Trim())")
            [void]$sb.AppendLine()
        }
        if ($n -eq 0) { continue }
        $all.Add([ordered]@{
            instruction = Fill (Pick $mathTemplates) $n
            input       = ""
            output      = $sb.ToString().TrimEnd().Replace("`r`n", "`n")
        })
    }
}

if ($all.Count -eq 0) { throw "No examples were created. Check your internet connection and try again." }

# ---------- Deduplicate, shuffle, write ----------
Write-Step "Writing $OutFile"
$seen  = [System.Collections.Generic.HashSet[string]]::new()
$lines = foreach ($ex in ($all | Sort-Object { $rng.Next() })) {
    if ($seen.Add($ex.output)) { $ex | ConvertTo-Json -Compress -Depth 3 }
}

$fullPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutFile)
$dir = Split-Path $fullPath -Parent
if ($dir) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
[System.IO.File]::WriteAllLines($fullPath, [string[]]$lines, [System.Text.UTF8Encoding]::new($false))

$mb = [Math]::Round((Get-Item $fullPath).Length / 1MB, 2)
Write-Host ""
Write-Host "Done: $(@($lines).Count) training examples, $mb MB  ->  $fullPath" -ForegroundColor Green
Write-Host "Rows downloaded from the internet: $totalDownloaded"
Write-Host "Next: commit data/train.jsonl (it's small) and train with:  python train.py --data data/train.jsonl"
