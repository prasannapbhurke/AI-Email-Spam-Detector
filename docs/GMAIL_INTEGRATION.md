# Gmail Integration Setup Guide

This guide walks through setting up Gmail API integration to automatically check incoming emails for spam.

---

## Prerequisites

- Google account with Gmail
- Python environment with all dependencies installed (`pip install -r requirements.txt`)
- API server running (this project) on http://localhost:8000

---

## Step 1: Get Google OAuth Credentials

### A. Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click project dropdown → **New Project**
3. Name: `Email Spam Detector` (or any name)
4. Click **Create**

### B. Enable Gmail API

1. In your project, go to **APIs & Services → Library**
2. Search for **Gmail API**
3. Click it → **Enable**

### C. Create OAuth 2.0 Credentials

1. Go to **APIs & Services → Credentials**
2. Click **Create Credentials** → **OAuth client ID**
3. Application type: **Desktop app**
4. Name: `Spam Detector Desktop`
5. Click **Create**
6. Click **Download JSON** → save as `credentials.json`
7. Copy `credentials.json` to the project root: `email-spam-detector/credentials.json`

**File structure:**
```
email-spam-detector/
├── credentials.json    ← place it here
├── gmail_token.json    ← will be created after auth
├── .env
└── ...
```

---

## Step 2: Configure Environment Variables

Edit `.env` file and add:

```env
# Gmail settings
GMAIL_CREDENTIALS_PATH=credentials.json
GMAIL_TOKEN_PATH=gmail_token.json
GMAIL_CHECK_INTERVAL=300
GMAIL_LABEL_SPAM=SPAM
GMAIL_LABEL_SAFE=INBOX

# Prediction API
PREDICTION_API_URL=http://localhost:8000
```

---

## Step 3: First Authentication

Run the Gmail monitor for the first time:

```bash
cd email-spam-detector
python scripts/run_gmail_spam_detector.py --credentials credentials.json
```

**What happens:**
1. A browser window opens asking you to sign in to Google
2. Grant permission to read/modify your Gmail
3. After approval, you're redirected to a page (may show error, that's OK)
4. Copy the full URL from browser address bar and paste in terminal
5. Token is saved to `gmail_token.json` for future use

**Tip:** If browser doesn't open automatically, copy the URL from terminal and open manually.

---

## Step 4: Test Run (Dry-Run)

First, test without modifying emails:

```bash
python scripts/run_gmail_spam_detector.py \
  --credentials credentials.json \
  --dry-run \
  --max-emails 10
```

This will:
- Fetch 10 most recent unread emails
- Send each to your spam detection API
- Print results without changing anything in Gmail

**Output example:**
```
======================================================================
PROCESSING RESULTS
======================================================================
  Total processed: 10
  Marked as spam:  3
  Marked as safe:   7
  Uncertain:        0
  Errors:           0
  Time taken:       2.34s
======================================================================
```

---

## Step 5: Run for Real (Optional)

Once you're confident, run without `--dry-run`:

```bash
# Process 50 emails and actually move spam to spam folder
python scripts/run_gmail_spam_detector.py \
  --credentials credentials.json \
  --max-emails 50 \
  --spam-threshold 0.75
```

**What it does:**
- Emails with spam probability ≥ 0.75 → move to `SPAM` label (archive from inbox)
- Emails with spam probability < 0.75 → keep in inbox (mark as read)

---

## Step 6: Continuous Monitoring (Optional)

Run continuously to check every 5 minutes:

```bash
python scripts/run_gmail_spam_detector.py \
  --credentials credentials.json \
  --continuous \
  --interval 300
```

Press `CTRL+C` to stop.

---

## How It Works

```
┌─────────────┐
│   Gmail     │ ← Fetch unread emails (Gmail API)
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Extract Text    │ ← Remove quotes, signatures, headers
│ & Clean         │
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│ Spam Detection   │ ← POST /predict (your trained model)
│ API (local)      │
└────────┬─────────┘
          │
          ▼
┌──────────────────┐
│ Action           │ ← Based on prediction
│ • Spam (≥0.7)    │   → Apply SPAM label + archive
│ • Safe (<0.7)    │   → Mark as read, keep in inbox
└──────────────────┘
```

---

## Important Security Notes

- **Never commit `credentials.json` or `gmail_token.json`** to git
- These files are already in `.gitignore`
- The token grants access to your Gmail - keep it secure
- Revoke access from Google Account settings if needed

---

## Troubleshooting

### "Authentication failed" or "Invalid grant"

- Delete `gmail_token.json` and re-run to get fresh token
- Ensure redirect URI in Google Cloud Console is `http://localhost` (for desktop app, auto-set)

### "Gmail API not enabled"

- Go to Google Cloud Console → APIs & Services → Library
- Search "Gmail API" → Enable

### "Insufficient Permission"

- The OAuth scope needs `https://mail.google.com/`
- Delete credentials and create new Desktop OAuth client
- Re-download credentials.json

### No emails processed

- Check you have unread emails in Gmail
- Use `--max-emails` to limit scope
- Verify API is running: `curl http://localhost:8000/health`

### Emails not moving

- In dry-run mode, nothing changes (use `--dry-run` flag)
- Gmail labels must exist (run with `--setup-labels` once)
- Check spam_threshold (default 0.7)

---

## Customization

### Change Thresholds

```bash
--spam-threshold 0.85  # More conservative (less spam)
--spam-threshold 0.60  # More aggressive (more spam)
```

### Process Specific Labels

By default, processes `INBOX` unread emails. To process other labels:

```bash
# Not implemented yet - would need to modify processor.py
```

### Custom API URL

```bash
--prediction-url https://your-cloud-api.com
```

---

## What the Script Does NOT Do

- ❌ Read already-read emails (only unread)
- ❌ Delete emails (only labels)
- ❌ Send emails
- ❌ Access other Gmail accounts without separate tokens
- ❌ Store email content long-term (only metadata in DB if enabled)

---

## Uninstalling / Cleanup

1. Revoke token: Google Account → Security → Third-party access
2. Delete `gmail_token.json`
3. Remove labels applied by the script (SPAM label in Gmail)

---

## Support

- Issues: https://github.com/prasannapbhurke/AI-Email-Spam-Detector/issues
- Gmail API docs: https://developers.google.com/gmail/api

---

*Last updated: April 17, 2026*
