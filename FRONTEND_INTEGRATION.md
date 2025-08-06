# Frontend Integration Guide - Chat Storage System

This document provides integration instructions for the new simplified chat storage system.

## Overview

The new storage system uses a single **auto-persist mode** with:
- Automatic conversation persistence with configurable retention policies
- User-controlled session management (no automatic session clearing on reconnect)
- Manual session saving for important conversations

## WebSocket Message Types

### 1. Clear History (Start New Session)

**Purpose**: User manually starts a new conversation session

**Message Format**:
```json
{
  "type": "clear_history"
}
```

**Frontend Implementation**:
```javascript
// Clear history button click handler
function clearHistory() {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({
            "type": "clear_history"
        }));
        
        // Clear local chat UI
        clearChatMessages();
        
        // Optionally show confirmation
        showNotification("New session started");
    }
}

// Example button HTML
// <button onclick="clearHistory()" class="clear-btn">🗑️ Clear History</button>
```

### 2. Save Session (Future Implementation)

**Purpose**: Save current conversation for future reference

**Message Format**:
```json
{
  "type": "save_session",
  "name": "My Important Conversation",  // Optional - auto-generated if not provided
  "conversation_id": "current-conv-id"  // Current conversation ID
}
```

**Frontend Implementation**:
```javascript
// Save session dialog
function saveCurrentSession() {
    const sessionName = prompt("Enter a name for this session (optional):");
    
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        const message = {
            "type": "save_session",
            "conversation_id": getCurrentConversationId()
        };
        
        if (sessionName && sessionName.trim()) {
            message.name = sessionName.trim();
        }
        
        websocket.send(JSON.stringify(message));
        showNotification("Session saved!");
    }
}
```

### 3. List Saved Sessions (Future Implementation)

**Purpose**: Get list of previously saved sessions

**Message Format**:
```json
{
  "type": "list_saved_sessions"
}
```

**Expected Response**:
```json
{
  "type": "saved_sessions_list",
  "sessions": [
    {
      "id": "session-uuid",
      "name": "My Important Conversation",
      "created_at": "2025-01-08T05:30:00Z",
      "message_count": 15,
      "session_start": "2025-01-08T05:15:00Z",
      "session_end": "2025-01-08T05:45:00Z"
    }
  ]
}
```

### 4. Load Saved Session (Future Implementation)

**Purpose**: Load a previously saved session

**Message Format**:
```json
{
  "type": "load_session",
  "session_id": "session-uuid-here"
}
```

## UI Implementation Examples

### 1. Clear History Button

```html
<!-- Recommended placement: Chat header or toolbar -->
<div class="chat-controls">
    <button id="clearHistoryBtn" class="btn-clear" title="Start New Conversation">
        <span class="icon">🔄</span>
        <span class="text">New Chat</span>
    </button>
</div>
```

```css
.btn-clear {
    background: #ff4444;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
}

.btn-clear:hover {
    background: #cc3333;
}
```

```javascript
document.getElementById('clearHistoryBtn').addEventListener('click', function() {
    // Optional: Show confirmation dialog
    if (confirm('Start a new conversation? Current chat will be saved to history.')) {
        clearHistory();
    }
});
```

### 2. Session Management (Future)

```html
<!-- Session management dropdown -->
<div class="session-controls">
    <button id="sessionMenuBtn" class="btn-session">
        <span class="icon">💾</span>
        <span class="text">Sessions</span>
    </button>
    
    <div id="sessionDropdown" class="dropdown hidden">
        <div class="dropdown-item" onclick="saveCurrentSession()">
            💾 Save Current Session
        </div>
        <div class="dropdown-item" onclick="loadSessionDialog()">
            📂 Load Saved Session
        </div>
        <hr>
        <div class="dropdown-item" onclick="clearHistory()">
            🔄 New Chat
        </div>
    </div>
</div>
```

### 3. WebSocket Message Handling

```javascript
// WebSocket message handler
websocket.onmessage = function(event) {
    const message = JSON.parse(event.data);
    
    switch (message.type) {
        case 'session_cleared':
            // Handle new session confirmation
            clearChatMessages();
            showNotification('New conversation started');
            break;
            
        case 'session_saved':
            // Handle session save confirmation
            showNotification(`Session saved: ${message.name}`);
            break;
            
        case 'saved_sessions_list':
            // Handle saved sessions list
            displaySavedSessions(message.sessions);
            break;
            
        case 'session_loaded':
            // Handle loaded session
            loadChatMessages(message.messages);
            showNotification(`Loaded session: ${message.session_name}`);
            break;
            
        // ... other message types
    }
};
```

## Configuration Integration

The frontend should respect the backend configuration for user experience:

```javascript
// Example: Check if session saving is enabled
const CONFIG = {
    sessionSavingEnabled: true,  // Based on backend saved_sessions.enabled
    maxSavedSessions: 50,       // Based on backend saved_sessions.max_saved
    retentionHours: 24          // Based on backend retention.max_age_hours
};

// Conditionally show session features
if (CONFIG.sessionSavingEnabled) {
    document.getElementById('saveSessionBtn').style.display = 'block';
} else {
    document.getElementById('saveSessionBtn').style.display = 'none';
}
```

## Best Practices

### 1. User Experience

- **Always confirm** before clearing history (unless user explicitly opts out)
- **Show visual feedback** when operations complete
- **Handle errors gracefully** with user-friendly messages
- **Preserve scroll position** when loading saved sessions

### 2. Performance

- **Debounce** session save requests to prevent spam
- **Cache** saved sessions list locally to avoid repeated requests
- **Lazy load** old messages when loading large saved sessions

### 3. Accessibility

```html
<!-- Accessible button examples -->
<button 
    id="clearHistoryBtn"
    aria-label="Start new conversation"
    title="Start new conversation - current chat will be saved to history"
>
    New Chat
</button>
```

## Error Handling

```javascript
// Handle WebSocket errors related to storage
websocket.onmessage = function(event) {
    const message = JSON.parse(event.data);
    
    if (message.type === 'error') {
        switch (message.error_code) {
            case 'STORAGE_FULL':
                showError('Storage full - please contact administrator');
                break;
            case 'SESSION_NOT_FOUND':
                showError('Saved session not found - it may have expired');
                break;
            case 'SAVE_DISABLED':
                showError('Session saving is currently disabled');
                break;
            default:
                showError(message.error_message || 'An error occurred');
        }
    }
};
```

## Implementation Phases

### Phase 1 (Current)
- ✅ Implement "Clear History" functionality
- ✅ Auto-persistence works automatically (no frontend changes needed)

### Phase 2 (Future)
- ⏳ Add "Save Session" functionality
- ⏳ Add "Load Session" functionality  
- ⏳ Add session management UI

### Phase 3 (Future Enhancements)
- ⏳ Search saved sessions
- ⏳ Session tags/categories
- ⏳ Export/import sessions

---

## Quick Start Checklist

- [ ] Add clear history button to your chat UI
- [ ] Implement `clearHistory()` function with WebSocket message
- [ ] Add confirmation dialog for clear history action
- [ ] Handle `session_cleared` response message
- [ ] Test with your existing chat application
- [ ] Plan for future session save/load features

The storage system is now ready for the "Clear History" functionality. Future session management features can be added incrementally without breaking existing functionality.
