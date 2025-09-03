# Streamlit Security Implementation Guide

## Overview

This document outlines the comprehensive security measures implemented for file uploads in the AIAgents4Pharma Streamlit applications. The security implementation addresses file validation, content scanning, and protection against various attack vectors.

## Security Architecture

### Multi-Layer Validation

The security implementation uses a multi-layer approach:

1. **File Extension Validation** - Whitelist approach for allowed file types
2. **MIME Type Verification** - Detects file masquerading attacks
3. **File Size Limits** - Prevents DoS attacks and resource exhaustion
4. **Content Pattern Scanning** - Detects malicious content patterns
5. **Filename Sanitization** - Prevents directory traversal attacks

## Implementation Details

### Core Security Functions

#### `secure_file_upload()`

The main security wrapper that replaces `st.file_uploader()`:

```python
from app.frontend.utils.streamlit_utils import secure_file_upload

# Basic usage
uploaded_file = secure_file_upload(
    "Upload Document",
    allowed_types=["pdf"],
    help_text="Upload a PDF document",
    max_size_mb=50,
    accept_multiple_files=False
)
```

#### `validate_uploaded_file()`

Comprehensive validation engine that performs:
- File extension checks against whitelist
- MIME type detection using `python-magic`
- File size validation
- Content pattern scanning
- Security threat detection

#### `sanitize_filename()`

Filename sanitization to prevent:
- Directory traversal attacks (`../../../etc/passwd`)
- Dangerous characters in filenames
- Overly long filenames
- Reserved system names

### Supported File Types

#### PDF Files
```python
pdf_file = secure_file_upload(
    "Upload Research Paper",
    allowed_types=["pdf"],
    max_size_mb=50
)
```
- **Extensions**: `.pdf`
- **MIME Types**: `application/pdf`
- **Content Validation**: Checks for PDF header (`%PDF-`)
- **Max Size**: 50MB (configurable)

#### XML/SBML Files
```python
model_file = secure_file_upload(
    "Upload Model File",
    allowed_types=["xml"],
    max_size_mb=25
)
```
- **Extensions**: `.xml`, `.sbml`
- **MIME Types**: `application/xml`, `text/xml`
- **Content Validation**: Checks for XML header (`<?xml`)
- **Max Size**: 25MB (configurable)

#### Spreadsheet Files
```python
data_file = secure_file_upload(
    "Upload Data File",
    allowed_types=["spreadsheet"],
    max_size_mb=25
)
```
- **Extensions**: `.xlsx`, `.xls`, `.csv`
- **MIME Types**: Excel/CSV MIME types
- **Max Size**: 25MB (configurable)

#### Text Files
```python
text_file = secure_file_upload(
    "Upload Text Data",
    allowed_types=["text"],
    max_size_mb=10
)
```
- **Extensions**: `.txt`, `.md`
- **MIME Types**: `text/plain`, `text/markdown`
- **Max Size**: 10MB (configurable)

## Security Configuration

### Upload Limits

```python
UPLOAD_SECURITY_CONFIG = {
    "max_file_size_mb": 50,  # Global default
    "max_filename_length": 255,
    "allowed_extensions": {
        "pdf": ["pdf"],
        "xml": ["xml", "sbml"],
        "spreadsheet": ["xlsx", "xls", "csv"],
        "text": ["txt", "md"],
    },
    "dangerous_extensions": [
        "exe", "bat", "cmd", "com", "pif", "scr", "vbs",
        "js", "jar", "app", "deb", "pkg", "dmg", "rpm",
        "msi", "dll", "sys", "drv", "sh", "bash", "ps1",
        "py", "pl", "rb", "php", "asp", "jsp"
    ]
}
```

### Blocked Content Patterns

The system automatically blocks files containing:
- Script tags: `<script>`, `javascript:`, `vbscript:`
- Server-side code: `<?php>`, `#!/bin/`
- Dangerous functions: `eval()`, `exec()`, `system()`
- Shell commands: `#!/usr/bin/`, `shell_exec()`

**Note**: The pattern `<%` is only blocked in non-PDF files, as it's part of legitimate PDF syntax. For PDFs, only truly suspicious patterns like `<% eval` or `<% system` are blocked.

## Application Integration

### Talk2AIAgents4Pharma (Combined) — T2AA4P

T2AA4P integrates both T2B and T2KG secure uploads via shared utilities, so no additional per-app code is needed beyond calling the helpers.

```python
# In app/frontend/streamlit_app_talk2aiagents4pharma.py

# SBML/PDF (T2B side)
uploaded_sbml_file = streamlit_utils.get_t2b_uploaded_files(app)

# Data packages and multimodal files (T2KG side)
streamlit_utils.get_uploaded_files(cfg)

# Both helpers use secure_file_upload() under the hood.
```

### Talk2BioModels - XML/SBML Upload

```python
def get_t2b_uploaded_files(app):
    uploaded_sbml_file = secure_file_upload(
        "Upload an XML/SBML file",
        allowed_types=["xml"],
        help_text="Upload a QSP as an XML/SBML file",
        max_size_mb=25,
        accept_multiple_files=False,
        key="secure_sbml_upload"
    )

    article = secure_file_upload(
        "Upload an article",
        allowed_types=["pdf"],
        help_text="Upload a PDF article to ask questions.",
        max_size_mb=50,
        accept_multiple_files=False,
        key="secure_article_upload"
    )
```

### Talk2KnowledgeGraphs - Data Upload

```python
def get_uploaded_files(cfg):
    data_package_files = secure_file_upload(
        "💊 Upload pre-clinical drug data",
        allowed_types=["text", "spreadsheet", "pdf"],
        help_text="Drug targets and kinetic parameters",
        max_size_mb=25,
        accept_multiple_files=True,
        key="secure_data_upload"
    )

    multimodal_files = secure_file_upload(
        "📦 Upload multimodal data package",
        allowed_types=["spreadsheet"],
        help_text="Multimodal endotype/phenotype data",
        max_size_mb=50,
        accept_multiple_files=True,
        key="secure_multimodal_upload"
    )
```

## Security Validation Flow

### 1. Pre-Upload Validation
- File extension whitelist check
- File type restriction by Streamlit

### 2. Post-Upload Validation
```python
validation_result = validate_uploaded_file(uploaded_file, allowed_types, max_size_mb)

if not validation_result["valid"]:
    st.error(f"❌ {uploaded_file.name}: {validation_result['error']}")
    return None

# Show warnings for suspicious but not critical issues
if validation_result["warnings"]:
    for warning in validation_result["warnings"]:
        st.warning(f"⚠️ {uploaded_file.name}: {warning}")

st.success(f"✅ {uploaded_file.name} validated successfully")
```

### 3. Secure Processing
```python
# Sanitize filename
safe_filename = sanitize_filename(uploaded_file.name)

# Create secure temporary file
with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{safe_filename}") as f:
    f.write(uploaded_file.read())
    secure_file_path = f.name

# Process using sanitized path
process_file(secure_file_path, safe_filename)
```

## Threat Protection

### File Masquerading Protection

The system detects files that have been renamed to bypass security:

```
malware.exe → research_paper.pdf  ❌ BLOCKED
- File extension: .pdf (appears safe)
- MIME type: application/x-executable (actual content)
- Result: BLOCKED with warning about MIME type mismatch
```

### Directory Traversal Protection

```python
# Dangerous filename examples (all blocked/sanitized):
"../../../etc/passwd"           → "etc_passwd"
"..\\windows\\system32\\cmd.exe" → "cmd.exe"
"/var/log/sensitive.log"        → "sensitive.log"
```

### Content Injection Protection

Files containing these patterns are automatically rejected:
- HTML/JavaScript: `<script>alert('xss')</script>`
- PHP code: `<?php system($_GET['cmd']); ?>`
- Shell commands: `#!/bin/bash rm -rf /`
- Python code injection: `eval(malicious_code)`

### Size-based DoS Protection

```python
# File size limits by type
PDF_MAX_SIZE = 50 * 1024 * 1024    # 50MB
XML_MAX_SIZE = 25 * 1024 * 1024    # 25MB
DATA_MAX_SIZE = 25 * 1024 * 1024   # 25MB
TEXT_MAX_SIZE = 10 * 1024 * 1024   # 10MB
```

## Error Handling

### Validation Errors

```python
# File too large
"File too large (75.2MB). Max: 50MB"

# Wrong file type
"File extension 'exe' not allowed. Allowed: ['pdf']"

# Dangerous content
"File contains suspicious content pattern: <script"

# MIME type mismatch (warning, not error)
"MIME type mismatch: detected 'application/x-executable', expected 'application/pdf'"
```

### User Feedback

The system provides clear visual feedback:
- ✅ **Success**: File validated and accepted
- ⚠️ **Warning**: File accepted but with concerns
- ❌ **Error**: File rejected with specific reason

## Best Practices

### For Developers

1. **Always use `secure_file_upload()`** instead of `st.file_uploader()`
2. **Sanitize filenames** before storing or processing
3. **Use appropriate file type restrictions** - only allow what's needed
4. **Set reasonable size limits** based on expected use cases
5. **Handle validation errors gracefully** with user-friendly messages

### For Users

1. **Upload only necessary file types** as specified
2. **Keep file sizes reasonable** (under the specified limits)
3. **Use descriptive, clean filenames** without special characters
4. **Verify file content matches the extension** before uploading

## Monitoring and Logging

### Security Events

The system logs security-related events:
- File validation failures
- MIME type mismatches
- Suspicious content detection
- Size limit violations

### Metrics

Key security metrics to monitor:
- Upload rejection rate
- Common rejection reasons
- File type distribution
- Size distribution

## Dependencies

### Required Packages

```toml
# pyproject.toml
dependencies = [
    "streamlit>=1.41.1",
    "python-magic>=0.4.27",  # MIME type detection
    # ... other dependencies
]
```

### System Requirements

- **python-magic**: Requires `libmagic` system library
  - **Linux**: `sudo apt-get install libmagic1`
  - **macOS**: `brew install libmagic`
  - **Windows**: Bundled with python-magic-bin

**Important**: Install `libmagic` before running Streamlit apps, or you'll get:
```
ImportError: failed to find libmagic. Check your installation
```

## Testing

### Security Test Cases

1. **Valid Files**: Ensure proper files are accepted
2. **File Masquerading**: Test renamed malicious files
3. **Size Limits**: Test files exceeding size limits
4. **Content Injection**: Test files with malicious patterns
5. **Directory Traversal**: Test dangerous filenames

### Example Tests

```python
def test_file_validation():
    # Test valid PDF
    valid_pdf = create_test_pdf()
    result = validate_uploaded_file(valid_pdf, ["pdf"])
    assert result["valid"] == True

    # Test file masquerading
    fake_pdf = create_executable_named_pdf()
    result = validate_uploaded_file(fake_pdf, ["pdf"])
    assert len(result["warnings"]) > 0  # Should warn about MIME mismatch

    # Test size limit
    large_file = create_large_file(100 * 1024 * 1024)  # 100MB
    result = validate_uploaded_file(large_file, ["pdf"], max_size_mb=50)
    assert result["valid"] == False
    assert "File too large" in result["error"]
```

## Future Enhancements

### Planned Improvements

1. **Virus Scanning Integration** - Add ClamAV or similar
2. **Advanced Content Analysis** - Deep content inspection
3. **User-based Quotas** - Per-user upload limits
4. **Audit Logging** - Enhanced security event logging
5. **Rate Limiting** - Prevent upload spam
6. **File Quarantine** - Temporary isolation of suspicious files

### Configuration Improvements

1. **Dynamic Configuration** - Runtime security policy updates
2. **Per-Agent Policies** - Different security rules per agent
3. **Content-based Rules** - Smarter content analysis
4. **Integration APIs** - External security service integration

---

## Conclusion

The Streamlit security implementation provides comprehensive protection against file upload attacks while maintaining usability. The multi-layer approach ensures that even if one security measure is bypassed, others will catch potential threats.

For questions or security concerns, please contact the development team or create an issue in the repository.

**Security is everyone's responsibility - always validate, never trust user input!** 🛡️
