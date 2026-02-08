---
name: meeting-recorder
description: Record and transcribe meetings with calendar integration, Salesforce account context, UCO tracking, and AI summaries with recommended UCO updates
---

# Meeting Recorder Skill

Record meetings with real-time transcription, speaker diarization, and AI-generated summaries that integrate with Salesforce Use Case Objects (UCOs).

## Prerequisites

- `stt` binary in PATH
- ANTHROPIC_API_KEY environment variable
- Screen Recording permission granted
- Google Calendar access (google-calendar skill)
- Salesforce access (salesforce-actions skill)
- Vibe profile at `~/.vibe/profile`

## Installation (if stt not found)

```bash
# Check if installed
which stt

# If not found, install:
which cargo || brew install rust
git clone https://github.com/bryan-db/meeting-notes.git ~/meeting-notes
cd ~/meeting-notes
cargo install --path .
```

## Instructions

### Step 1: Get Meeting from Calendar

Use the **google-calendar** skill to find the current or upcoming meeting:

```
Find meetings starting within the next 15 minutes or currently in progress.
Extract:
- Meeting title
- Start/end time
- Attendees (email addresses)
- Description/agenda
- Google Meet/Zoom link (if present)
```

If no calendar access, ask the user for meeting details.

### Step 2: Load Vibe Profile

Read the SA's vibe profile to understand their accounts:

```bash
cat ~/.vibe/profile
```

Extract from the profile:
- SA's name and email
- List of accounts they own
- Account contacts and domains

### Step 3: Match Attendees to Account

Compare meeting attendees against vibe profile accounts:

1. Extract email domains from attendees (e.g., `alice@acme.com` → `acme.com`)
2. Match domains against account domains in vibe profile
3. Identify the relevant Salesforce Account

**Matching logic:**
```
For each attendee email:
  - Skip @databricks.com (internal)
  - Extract domain
  - Search vibe profile accounts for matching domain
  - If found, this is the customer account
```

If no match found, ask the user which account this meeting is for.

### Step 4: Fetch Open UCOs from Salesforce

Use the **salesforce-actions** skill to get open Use Case Objects for the matched account:

```
Get all Use Case Objects for Account: [Account Name]
Filter: Status NOT IN ('Closed - Won', 'Closed - Lost', 'Closed - No Decision')
Include:
- UCO Name
- Stage
- Use Case Type
- Last Activity Date
- Recent Notes/Updates
- Owner
- Target Close Date
```

Format UCOs as a list for context.

### Step 5: Create Meeting Folder

Create folder at `~/Documents/meetings/` with format:
```
YYYY-MM-DD_<account-name>_<meeting-title>/
```

Example: `2025-02-05_acme-corp_technical-deep-dive/`

```bash
mkdir -p ~/Documents/meetings/<folder-name>
```

### Step 6: Create CONTEXT.md (CRITICAL)

Write comprehensive context to `<meeting-folder>/CONTEXT.md`:

```markdown
# Meeting Context

## Meeting Details
- **Title**: [From calendar]
- **Date/Time**: [Start - End time]
- **Duration**: [X minutes]

## Account Information
- **Account Name**: [From Salesforce]
- **Account ID**: [Salesforce ID]
- **Industry**: [If available]
- **SA Owner**: [From vibe profile]

## Attendees

### Customer
- [Name] <email> - [Title if known]
- [Name] <email>

### Databricks
- [Name] <email> - [Role]

## Meeting Purpose
[From calendar description or user input]

## Agenda
[From calendar or user input]

## Open Use Case Objects

### UCO: [UCO Name 1]
- **Stage**: [Current stage]
- **Type**: [Use case type]
- **Target Close**: [Date]
- **Last Update**: [Date] - [Summary of last note]
- **Key Details**: [Relevant context]

### UCO: [UCO Name 2]
- **Stage**: [Current stage]
- **Type**: [Use case type]
- **Target Close**: [Date]
- **Last Update**: [Date] - [Summary of last note]

## Recent Account Activity
[Any recent notes, meetings, or updates from Salesforce]

## Background Context
[Any additional context the SA provides]
```

### Step 7: Create PROMPT.md for UCO-Aware Summarization

Write to `<meeting-folder>/PROMPT.md`:

```markdown
You are summarizing a customer meeting for a Databricks Solutions Architect.

## Required Output Sections

### 1. Meeting Overview
Brief description of meeting purpose, key attendees, and overall outcome.

### 2. Key Discussion Points
Main topics discussed with relevant technical details.

### 3. Customer Priorities & Pain Points
What the customer cares about most, concerns raised, and challenges mentioned.

### 4. Technical Details
Any technical requirements, architecture discussions, data volumes, timelines mentioned.

### 5. Decisions Made
Any decisions or agreements reached during the meeting.

### 6. Action Items
Extract ALL action items with:
- [ ] **Task** - Owner - Due date (if mentioned)

Look for: "I'll", "we need to", "can you", "let's", "follow up", "next steps", "action item", "TODO"

### 7. Recommended UCO Updates

Based on the meeting discussion and the Open UCOs listed in the context, recommend specific updates:

For each relevant UCO:
```
**UCO: [Name]**
- Recommended Stage Change: [If applicable, e.g., "Move from Discovery to Technical Win"]
- Update Notes: [Specific notes to add based on meeting discussion]
- Key Takeaways: [What was learned relevant to this use case]
```

If a NEW use case was discussed that doesn't match existing UCOs:
```
**NEW UCO Recommended**
- Use Case Type: [e.g., Lakehouse, ML, Real-time, etc.]
- Description: [What the customer wants to achieve]
- Initial Stage: Discovery
- Notes: [Context from meeting]
```

### 8. Next Steps
Planned follow-ups, next meeting, POC timeline, etc.

### 9. Risk Flags
Any concerns, blockers, or competitive mentions to be aware of.

## Formatting
- Use markdown formatting
- Be specific and actionable
- Include technical details when relevant
- Reference specific attendees by name when attributing statements
```

### Step 8: Start Recording

Launch stt in a **new terminal window** so the user can see the TUI:

**For Ghostty:**
```bash
open -na Ghostty.app --args -e "stt meeting --output ~/Documents/meetings/<folder-name>"
```

**For Terminal.app:**
```bash
osascript -e 'tell application "Terminal" to do script "stt meeting --output ~/Documents/meetings/<folder-name>"'
```

**For iTerm2:**
```bash
osascript -e 'tell application "iTerm" to create window with default profile command "stt meeting --output ~/Documents/meetings/<folder-name>"'
```

**Tell the user:**
```
Meeting recorder started in a new terminal window.

Keyboard shortcuts:
- q : End meeting and generate summary
- m : Add marker (ACTION_ITEM, DECISION, BLOCKER)
- n : Add manual note
- s : Capture screenshot (region), S : screenshot (window)
- j/k or ↑/↓ : Scroll transcript, G/g : jump to bottom/top

When you're done, press 'q' to generate the summary with UCO recommendations.
```

### Step 9: After Meeting Ends

When the user indicates the meeting is over (or you detect the stt process has ended):

1. Read the generated `SUMMARY.md`
2. Present key findings to the user
3. Offer to help update UCOs in Salesforce

```
Meeting notes saved to: ~/Documents/meetings/<folder-name>/

## Summary Highlights
[Key points from SUMMARY.md]

## Recommended UCO Updates
[From the summary]

Would you like me to:
1. Update the UCOs in Salesforce with these notes?
2. Create a new UCO for [new use case if recommended]?
3. Send a follow-up email to attendees?
```

## Example Flow

```
User: "Record my meeting"

Agent:
1. [google-calendar] Get current meeting: "Acme Corp - Data Platform Review"
   - Attendees: alice@acme.com, bob@acme.com, sa@databricks.com
   - Time: 2:00 PM - 3:00 PM

2. [Read ~/.vibe/profile] SA owns Acme Corp account

3. [salesforce-actions] Get UCOs for Acme Corp:
   - "Lakehouse Migration" - Technical Win stage
   - "Real-time Analytics POC" - Discovery stage

4. Create folder: ~/Documents/meetings/2025-02-05_acme-corp_data-platform-review/

5. Write CONTEXT.md with meeting details + UCO info

6. Write PROMPT.md with UCO-aware summarization instructions

7. Launch: open -na Ghostty.app --args -e "stt meeting --output ..."

8. User conducts meeting, presses 'q' when done

9. Present summary with UCO update recommendations

10. Offer to update Salesforce UCOs
```

## TUI Keyboard Reference

| Key | Action |
|-----|--------|
| q | Quit and process meeting |
| m | Add marker (ACTION_ITEM, DECISION, BLOCKER, QUESTION) |
| n | Add manual note |
| s | Screenshot (region selection) |
| S | Screenshot (window selection) |
| j/↓ | Scroll down |
| k/↑ | Scroll up |
| G | Jump to bottom |
| g | Jump to top |

## Troubleshooting

- **stt not found**: Run installation steps above
- **Models not found**: Auto-download on first run (~500MB)
- **No audio**: Grant Screen Recording permission in System Settings > Privacy
- **No calendar access**: Use google-calendar skill to authenticate
- **No Salesforce access**: Use salesforce-actions skill to authenticate
- **No vibe profile**: Run `vibe profile` to create one
