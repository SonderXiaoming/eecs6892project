# run_all.ps1
# Batch runner for MiniGrid exploration experiments on Windows PowerShell.
#
# Quiet GPU test:
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -UseUv -Device cuda -Timesteps 10000 -Envs FourRooms -Strategies baseline -Seeds 0 -SkipHeatmap
#
# Fully quiet:
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -UseUv -Device cuda -NoProgress -Timesteps 10000 -Envs FourRooms -Strategies baseline -Seeds 0 -SkipHeatmap
#
# Full run:
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -UseUv -Device cuda -SkipHeatmap

# Envs = "FourRooms", "DoorKey", "MultiRoom"
# "baseline", "action_bonus", "state_bonus", "reward_shaping", "high_entropy"

param(
    [string[]]$Envs = @("DoorKey"),
    [string[]]$Strategies = @("baseline"),
    [int[]]$Seeds = @(42),
    [int]$Timesteps = 512000,
    [int]$Episodes = 100,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cpu",
    [int]$Verbose = 0,
    [switch]$UseUv,
    [switch]$NoProgress,
    [switch]$SkipTrain,
    [switch]$SkipEval,
    [switch]$SkipPlot,
    [switch]$SkipHeatmap
)

$ErrorActionPreference = "Stop"

function Run-PythonScript {
    param(
        [string]$Script,
        [string[]]$Arguments
    )

    if ($UseUv) {
        $cmd = "uv"
        $args = @("run", "python", $Script) + $Arguments
    }
    else {
        $cmd = "python"
        $args = @($Script) + $Arguments
    }

    Write-Host ""
    Write-Host ">>> $cmd $($args -join ' ')" -ForegroundColor Cyan

    & $cmd @args

    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $cmd $($args -join ' ')"
    }
}

Write-Host "MiniGrid Exploration Experiment Runner" -ForegroundColor Green
Write-Host "Envs:        $($Envs -join ', ')"
Write-Host "Strategies:  $($Strategies -join ', ')"
Write-Host "Seeds:       $($Seeds -join ', ')"
Write-Host "Timesteps:   $Timesteps"
Write-Host "Episodes:    $Episodes"
Write-Host "Device:      $Device"
Write-Host "Verbose:     $Verbose"
Write-Host "Progress:    $(if ($NoProgress) { 'off' } else { 'on' })"
Write-Host "Python mode: $(if ($UseUv) { 'uv run python' } else { 'python' })"

if (-not $SkipTrain) {
    Write-Host ""
    Write-Host "========== Training ==========" -ForegroundColor Yellow

    foreach ($envName in $Envs) {
        foreach ($strategy in $Strategies) {
            foreach ($seed in $Seeds) {
                $trainArgs = @(
                    "--env", $envName,
                    "--strategy", $strategy,
                    "--seed", "$seed",
                    "--timesteps", "$Timesteps",
                    "--device", "$Device",
                    "--verbose", "$Verbose"
                )

                if ($NoProgress) {
                    $trainArgs += "--no-progress"
                }

                Run-PythonScript "train.py" $trainArgs
            }
        }
    }
}
else {
    Write-Host "Skipping training." -ForegroundColor DarkYellow
}

if (-not $SkipEval) {
    Write-Host ""
    Write-Host "========== Evaluation ==========" -ForegroundColor Yellow

    foreach ($envName in $Envs) {
        foreach ($strategy in $Strategies) {
            foreach ($seed in $Seeds) {
                Run-PythonScript "evaluate.py" @(
                    "--env", $envName,
                    "--strategy", $strategy,
                    "--seed", "$seed",
                    "--episodes", "$Episodes"
                )
            }
        }
    }
}
else {
    Write-Host "Skipping evaluation." -ForegroundColor DarkYellow
}

if (-not $SkipPlot) {
    Write-Host ""
    Write-Host "========== Plotting ==========" -ForegroundColor Yellow

    Run-PythonScript "plot_results.py" @()

    if (-not $SkipHeatmap) {
        foreach ($envName in $Envs) {
            foreach ($strategy in $Strategies) {
                foreach ($seed in $Seeds) {
                    Run-PythonScript "plot_heatmap.py" @(
                        "--env", $envName,
                        "--strategy", $strategy,
                        "--seed", "$seed",
                        "--episodes", "20"
                    )
                }
            }
        }
    }
    else {
        Write-Host "Skipping heatmaps." -ForegroundColor DarkYellow
    }
}
else {
    Write-Host "Skipping plotting." -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "All selected tasks finished." -ForegroundColor Green
