# Deployment Update Issue - Root Cause and Fix

## Problem Summary
You edit code and expect the server to update automatically through GitHub Actions YAML, but updates stopped reaching the server.

## Why This Happened
I checked the repository and found these blockers:

1. **Branch trigger mismatch**
- Your deploy workflow was configured to run only on `main`.
- Your current work branch is `ninja-updates`.
- Result: pushes on `ninja-updates` did not trigger deployment.

2. **Hardcoded pull branch on server**
- The workflow always ran `git pull origin main` on the server.
- Even if another branch triggered deployment, server still pulled `main`.

3. **Potential missing dependency on runner**
- The workflow uses `sshpass` but did not install it explicitly.
- On some runner images this can fail and stop deployment.

## What I Changed
I updated `.github/workflows/deploy.yml` with the following fixes:

1. **Trigger deployment on both branches**
- Added:
  - `main`
  - `ninja-updates`

2. **Added manual run option**
- Added `workflow_dispatch` with optional `deploy_branch` input.

3. **Branch-aware deployment**
- Added a step to resolve `DEPLOY_BRANCH`:
  - Uses `workflow_dispatch` input if provided.
  - Otherwise uses the branch that triggered the workflow.

4. **Install `sshpass` before deploy step**
- Added `apt-get install -y sshpass`.

5. **Deploy the resolved branch on server**
- Replaced fixed `main` pull with:
  - `git fetch origin`
  - `git checkout "$DEPLOY_BRANCH"`
  - `git pull origin "$DEPLOY_BRANCH"`

6. **Minor modernization**
- Updated checkout action from `actions/checkout@v2` to `actions/checkout@v4`.

## Files Updated
- `.github/workflows/deploy.yml`

## How to Verify
1. Push a small commit to `ninja-updates`.
2. Open GitHub Actions and confirm `Deploy PhotoMaster Streamlit App` runs.
3. Confirm deploy logs include your branch name.
4. On server, verify new commit is present:
   - `cd /home/shobbak/superpower/App_v1`
   - `git rev-parse --abbrev-ref HEAD`
   - `git log -1 --oneline`
5. Check app availability after service restart.

## Important Notes
1. Server must already have the target branch available in the repo directory.
2. `SSH_PASSWORD` secret must exist in GitHub repository secrets.
3. `shobbak` user must be allowed to restart `PhMr.service` with sudo password.

## Recommended Next Improvement
Use SSH key-based deployment instead of password + `sshpass` for better security and reliability.
