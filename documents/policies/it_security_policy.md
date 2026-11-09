# IT Security Policy

> **Disclaimer:** This is a **fictional sample policy** created exclusively for software development and testing of the Internal Policy Q&A Service. It does not represent the policies of any real organization.

| Field | Value |
|-------|-------|
| **Policy ID** | POL-IT-001 |
| **Version** | 1.0 |
| **Effective date** | 2026-01-01 |
| **Last updated** | 2026-10-01 |
| **Owner** | IT Security Team |
| **Applies to** | All employees and contractors of Northwind Analytics (fictional) |
| **Review cycle** | Annual |

---

## 1. Purpose and Scope

This policy sets the minimum security requirements for accounts, devices, and company data. It applies to **all employees and contractors** and to all devices used to access company systems. Breaches of this policy may lead to loss of access and disciplinary action.

Security issues and requests are raised through the **IT Service Portal**.

## 2. Password Requirements

- Passwords must be **at least 14 characters** long.
- Passwords must include at least **three** of the following four: uppercase letters, lowercase letters, numbers, symbols.
- Passwords must **not** reuse any of the previous **10 passwords** and must not contain the username or the word "password".
- Passwords for standard accounts expire every **365 days**. Passwords suspected of compromise must be changed **immediately**.
- Passwords must **never be shared**, written down in plain text, or stored in unapproved locations. Employees must use the **company-approved password manager** to store credentials.
- Default or vendor-supplied passwords must be changed before a system is used.

## 3. Multi-Factor Authentication (MFA)

- **MFA is mandatory** for all company accounts and for all remote access to company systems.
- Approved MFA methods are, in order of preference:
  1. a hardware security key (FIDO2);
  2. an authenticator app (time-based one-time passcode);
  3. a push notification to a registered company device.
- **SMS-based codes** are permitted only as a temporary fallback and must not be the primary method.
- Employees must register at least **two** MFA methods so that a lost device does not lock them out.
- MFA prompts must never be approved unless the employee personally initiated the sign-in. Unexpected prompts must be **denied and reported** (Section 6).

## 4. Device Security

- All laptops and mobile devices that access company data must:
  - have **full-disk encryption** enabled;
  - run a **supported, patched** operating system with automatic security updates on;
  - run the **company-managed endpoint protection** agent;
  - lock automatically after **10 minutes** of inactivity and require authentication to unlock.
- Devices must not be left unattended and unlocked in public spaces.
- **Personal devices** may access email and calendar only if enrolled in **company mobile device management (MDM)**. Company data must not be stored on unmanaged personal devices or personal cloud accounts.
- Lost or stolen devices must be reported to the IT Service Portal **within 1 hour** of discovery so the device can be remotely locked or wiped.
- Software may only be installed from the **approved software catalog**; installing unapproved software is prohibited.

## 5. Data Handling

Data is classified into three levels:

| Classification | Examples | Handling |
|----------------|----------|----------|
| **Public** | Published marketing material | No restrictions |
| **Internal** | Internal policies, project plans | Share only within the company; do not post externally |
| **Confidential** | Personal data, financials, credentials, source code | Encrypt in transit and at rest; share only on a need-to-know basis via approved systems |

- **Confidential** data must only be sent using company-approved, encrypted channels. It must **not** be sent to personal email accounts or uploaded to unapproved services.
- Personal data must be handled in line with applicable data-protection law and only for legitimate business purposes.
- Printed confidential documents must be stored securely and shredded when no longer needed.
- Access to confidential systems is governed by the **Access Control Policy (POL-IT-002)**.

## 6. Reporting Security Incidents

- A security incident is any actual or suspected event that threatens the confidentiality, integrity, or availability of company systems or data — for example, phishing, malware, lost devices, or unexpected MFA prompts.
- Employees must report suspected incidents to the **IT Security Team via the IT Service Portal, or by phone for urgent cases, within 1 hour** of discovery.
- Employees must **not** attempt to investigate or "clean up" an incident themselves, as this can destroy evidence.
- Suspected **phishing emails** must be reported using the **"Report Phishing"** button and must not be forwarded to colleagues.
- All reports are treated confidentially. Reporting a genuine suspected incident in good faith will **never** result in penalty, even if it turns out to be a false alarm.

## 7. Related Policies

- Access Control Policy (POL-IT-002)
- Remote Work Policy (POL-HR-002)
