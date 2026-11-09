# Access Control Policy

> **Disclaimer:** This is a **fictional sample policy** created exclusively for software development and testing of the Internal Policy Q&A Service. It does not represent the policies of any real organization.

| Field | Value |
|-------|-------|
| **Policy ID** | POL-IT-002 |
| **Version** | 1.0 |
| **Effective date** | 2026-01-01 |
| **Last updated** | 2026-10-01 |
| **Owner** | IT Security Team |
| **Applies to** | All employees and contractors of Northwind Analytics (fictional) |
| **Review cycle** | Annual |

---

## 1. Purpose and Scope

This policy governs how access to company systems and data is requested, granted, reviewed, and revoked. It applies to all employees and contractors and complements the **IT Security Policy (POL-IT-001)**. Access is managed through the **IT Service Portal** and the central **identity and access management (IAM)** system.

## 2. Guiding Principles

- **Least privilege:** users receive the minimum access needed to perform their role, and no more.
- **Need to know:** access to confidential data is granted only where there is a legitimate business need.
- **Segregation of duties:** no single person should control an entire high-risk process end to end (for example, both creating and approving a payment).
- **Accountability:** every access grant has a named requester, approver, and business justification recorded in the IAM system.

## 3. Account Access Requests

- Access is requested through the **IT Service Portal**, specifying the system, the **role or permission set** required, and a business justification.
- For **new joiners**, the line manager raises a request **at least 3 working days before** the start date so that baseline access is ready on day one.
- **Baseline access** (email, collaboration tools, and the HR system) is provisioned automatically for all new joiners based on their role template.
- Requests for **additional or elevated access** beyond the role template must be raised individually and approved under Section 5.
- **Shared or generic accounts are prohibited.** Where a service account is genuinely required, it must have a named owner and must not be used for interactive login.

## 4. Role-Based Permissions

- Access is granted through **roles** (role-based access control, RBAC), not to individuals directly. Each role bundles the permissions needed for a job function.
- Each role has a designated **system owner** who is accountable for the permissions it contains.
- Standard role templates exist for common job functions (for example, "Engineer", "Finance Analyst", "People Team"). Role templates are reviewed at least **annually** by their system owners.
- **Privileged roles** (for example, system administrators and those who can access confidential data in bulk) require:
  - approval from the **system owner and the IT Security Team**;
  - a documented business justification;
  - MFA on every sign-in, as required by the IT Security Policy (POL-IT-001).

## 5. Access Approval

1. The request routes to the employee's **line manager**, who confirms the business need.
2. It then routes to the relevant **system owner**, who confirms the specific permissions are appropriate.
3. **Privileged or confidential-data access** additionally requires approval from the **IT Security Team**.
4. Each approver must **approve or reject within 3 working days**; rejections must state a reason. Requests unresolved after 3 working days are escalated to the system owner's manager.
5. Approvals are recorded in the IAM system with the approver, date, and justification. Access is only provisioned after all required approvals are recorded.
6. **Temporary access** (for example, for a short project) must specify an **expiry date of no more than 90 days**, after which it is revoked automatically unless renewed through the same process.

## 6. Periodic Access Reviews

- **System owners** review access to their systems at least **quarterly** and confirm that each grant is still required.
- **Privileged access** is reviewed **monthly**.
- Line managers review their team members' access as part of each access review and flag anything no longer needed.
- Access identified as unnecessary during a review must be **removed within 5 working days**.
- Each review is recorded in the IAM system, including the reviewer, date, and outcome. Reviews are retained for audit for at least **24 months**.

## 7. Employee Offboarding and Access Revocation

- The line manager or the People Team must notify IT of a leaver through the IT Service Portal **as soon as the leaving date is known**, and no later than the employee's **last working day**.
- **On the last working day:**
  - all interactive access (email, collaboration tools, business applications, and VPN) is **disabled by end of day**;
  - **privileged access is revoked within 2 hours** of the confirmed departure time.
- **Involuntary departures** (for example, dismissal) require **immediate revocation of all access**, coordinated between the People Team and the IT Security Team, effective at the time the employee is informed.
- Company devices must be returned by the last working day; see the IT Security Policy (POL-IT-001) for device handling and remote wipe.
- A **leaver's accounts are disabled, not deleted**, and are retained in a disabled state for **90 days** before deletion, to preserve records and allow for handover.
- A **post-departure access check** is completed within **5 working days** of the leaving date to confirm that no access remains active.

## 8. Related Policies

- IT Security Policy (POL-IT-001)
- Remote Work Policy (POL-HR-002)
- Expense Reimbursement Policy (POL-FIN-001)
