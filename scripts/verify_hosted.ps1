param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$BaseUrl
)

$ErrorActionPreference = 'Stop'
$BaseUrl = $BaseUrl.TrimEnd('/')

function Invoke-HostedCheck {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Uri,
        [ValidateSet('Get', 'Post')][string]$Method = 'Get',
        [string]$Body
    )

    Write-Output "=== $Name ==="
    try {
        $parameters = @{
            Uri = $Uri
            Method = $Method
            TimeoutSec = 45
            ErrorAction = 'Stop'
        }
        if ($Method -eq 'Post') {
            $parameters.ContentType = 'application/json'
            $parameters.Body = $Body
        }
        $response = Invoke-WebRequest @parameters -UseBasicParsing
        Write-Output ("HTTP_STATUS=" + [int]$response.StatusCode)
        if ($response.Content) {
            try {
                $response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 20
            }
            catch {
                Write-Output ("BODY=" + $response.Content)
            }
        }
        else {
            Write-Output 'BODY='
        }
    }
    catch {
        $httpResponse = $_.Exception.Response
        if ($httpResponse -and $httpResponse.StatusCode) {
            Write-Output ("HTTP_STATUS=" + [int]$httpResponse.StatusCode)
        }
        else {
            Write-Output 'HTTP_STATUS=unavailable'
        }
        if ($_.ErrorDetails.Message) {
            Write-Output ("BODY=" + $_.ErrorDetails.Message)
        }
        elseif ($httpResponse) {
            $stream = $httpResponse.GetResponseStream()
            if ($stream) {
                $reader = New-Object System.IO.StreamReader($stream)
                $body = $reader.ReadToEnd()
                $reader.Dispose()
                Write-Output ("BODY=" + $body)
            }
            else {
                Write-Output ("ERROR=" + $_.Exception.Message)
            }
        }
        else {
            Write-Output ("ERROR=" + $_.Exception.Message)
        }
    }
}

$supportedBody = @{ query = 'potholes on Mwangaza Road near Kifaru Market' } | ConvertTo-Json -Compress
$missingBody = @{ query = "What is the county's budget for road repairs this year?" } | ConvertTo-Json -Compress

Invoke-HostedCheck -Name 'GET /healthz' -Uri "$BaseUrl/healthz"
Invoke-HostedCheck -Name 'GET /status' -Uri "$BaseUrl/status"
Invoke-HostedCheck -Name 'Supported query: Mwangaza Road potholes' -Uri "$BaseUrl/process" -Method Post -Body $supportedBody
Invoke-HostedCheck -Name 'Missing evidence: county road-repair budget' -Uri "$BaseUrl/process" -Method Post -Body $missingBody
