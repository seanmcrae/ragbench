"""SYNTHETIC help-center corpus for "Tallyhive", a fictional invoicing and time-tracking SaaS.

Everything here is invented for evaluation purposes: the product, plans, limits, error codes
and policies do not describe any real service. Third-party product names appear only as
integration targets so the corpus reads like a real help center.

The corpus is assembled from authored fact tables and rendered with a seeded RNG, so the same
seed always yields byte-identical documents, queries, graded qrels and reference answers.
Facts are phrased as copular sentences ("X is Y") the way help-center articles usually are.

Query types (stored in ``Query.metadata["type"]``):
  plan_attribute, feature_location, feature_plan, feature_limit, error_fix, error_cause,
  integration_sync, integration_objects, policy  -- single-hop
  multi_hop  -- the answer document is only identifiable after resolving a bridge fact
Relevance grades: 2 = contains the answer, 1 = on-topic supporting or bridge document.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from rag_eval.datasets.base import Dataset, Document, Qrels, Query

PRODUCT = "Tallyhive"
PLANS: tuple[str, ...] = ("Starter", "Team", "Business", "Enterprise")
DEFAULT_SEED = 7


@dataclass(frozen=True)
class PlanAttribute:
    key: str
    title: str
    subject: str
    short: str
    values: tuple[str, str, str, str]
    context: tuple[str, ...]
    questions: tuple[str, ...]


@dataclass(frozen=True)
class Feature:
    name: str
    path: str
    min_plan: str
    limit_subject: str
    limit_value: str


@dataclass(frozen=True)
class Integration:
    name: str
    category: str
    interval: str
    objects: str
    min_plan: str


@dataclass(frozen=True)
class Policy:
    title: str
    fact: str
    answer: str
    context: tuple[str, ...]
    questions: tuple[str, ...]


@dataclass(frozen=True)
class ErrorKind:
    prefix: str
    title: str
    cause: str
    fix: str


PLAN_ATTRIBUTES: tuple[PlanAttribute, ...] = (
    PlanAttribute(
        "api-rate-limit",
        "API rate limits",
        "the API rate limit",
        "API rate limit",
        (
            "60 requests per minute",
            "300 requests per minute",
            "1200 requests per minute",
            "5000 requests per minute",
        ),
        (
            "Requests above the limit receive HTTP 429 with a Retry-After header.",
            "Limits are counted per workspace rather than per API token.",
        ),
        (
            "What is the API rate limit on the {plan} plan?",
            "How many API calls per minute can a {plan} workspace make?",
            "We keep getting 429 responses on {plan}. What is our request limit?",
        ),
    ),
    PlanAttribute(
        "storage",
        "File storage",
        "the file storage allowance",
        "file storage allowance",
        ("2 GB per workspace", "100 GB per workspace", "1 TB per workspace", "10 TB per workspace"),
        (
            "Receipts, attachments and exported reports all count toward storage.",
            "Owners receive an email when usage passes 90 percent of the allowance.",
        ),
        (
            "How much file storage does the {plan} plan include?",
            "What is the attachment storage cap for {plan} workspaces?",
        ),
    ),
    PlanAttribute(
        "audit-retention",
        "Audit log retention",
        "audit log retention",
        "audit log retention",
        ("7 days", "90 days", "1 year", "7 years"),
        (
            "The audit log records sign-ins, permission changes and invoice edits.",
            "Older entries are deleted automatically once they pass the retention period.",
        ),
        (
            "How long are audit logs kept on {plan}?",
            "What is the audit log retention period for the {plan} plan?",
        ),
    ),
    PlanAttribute(
        "seat-cap",
        "Seat limits",
        "the maximum number of seats",
        "seat cap",
        ("3", "25", "250", "unlimited"),
        (
            "Guests invited to the client portal do not use a seat.",
            "Deactivated members free their seat at the start of the next billing period.",
        ),
        (
            "How many users can a {plan} workspace have?",
            "What is the seat cap on {plan}?",
        ),
    ),
    PlanAttribute(
        "support-response",
        "Support response times",
        "the first-response target for support tickets",
        "support response target",
        ("72 hours", "24 hours", "4 business hours", "1 hour"),
        (
            "Tickets are answered in English, German and Spanish.",
            "Urgent outages should be reported through the in-app status banner.",
        ),
        (
            "How quickly does support respond on the {plan} plan?",
            "What is the support SLA for {plan} customers?",
        ),
    ),
    PlanAttribute(
        "price",
        "Pricing",
        "the list price",
        "list price",
        (
            "free for up to 3 seats",
            "12 dollars per seat per month",
            "29 dollars per seat per month",
            "quoted per contract",
        ),
        (
            "Annual billing applies a 15 percent discount to the monthly list price.",
            "Prices exclude sales tax and VAT.",
        ),
        (
            "How much does the {plan} plan cost?",
            "What is the per-seat price of {plan}?",
        ),
    ),
)

PLAN_AUDIENCE = {
    "Starter": "freelancers who invoice a handful of clients",
    "Team": "small agencies that track time across projects",
    "Business": "growing companies that need approvals and finance controls",
    "Enterprise": "organisations with security, compliance and procurement requirements",
}

FEATURES: tuple[Feature, ...] = (
    Feature(
        "recurring invoices",
        "Invoices > Recurring > New schedule",
        "Team",
        "the shortest billing interval for a recurring schedule",
        "one week",
    ),
    Feature(
        "multi-currency invoicing",
        "Settings > Billing > Currencies",
        "Team",
        "the number of supported invoice currencies",
        "38",
    ),
    Feature(
        "late payment reminders",
        "Invoices > Reminders",
        "Starter",
        "the maximum number of reminders per invoice",
        "5",
    ),
    Feature(
        "receipt scanning",
        "Expenses > Scan receipt",
        "Starter",
        "the largest receipt file accepted",
        "15 MB",
    ),
    Feature(
        "mileage tracking",
        "Expenses > Mileage",
        "Team",
        "the default mileage rate source",
        "the IRS standard rate for the current year",
    ),
    Feature(
        "approval workflows",
        "Settings > Approvals > Workflows",
        "Business",
        "the maximum number of approval steps",
        "6",
    ),
    Feature(
        "timesheets",
        "Time > Timesheets",
        "Starter",
        "the timesheet lock period",
        "14 days after the week ends",
    ),
    Feature(
        "project budgets",
        "Projects > Budget",
        "Team",
        "the default budget alert threshold",
        "80 percent of the budget",
    ),
    Feature(
        "the client portal",
        "Clients > Portal settings",
        "Team",
        "the lifetime of a client portal link",
        "30 days",
    ),
    Feature(
        "custom invoice templates",
        "Settings > Branding > Templates",
        "Team",
        "the number of custom templates allowed",
        "10",
    ),
    Feature(
        "SAML single sign-on",
        "Settings > Security > SAML SSO",
        "Business",
        "the supported identity provider protocol",
        "SAML 2.0",
    ),
    Feature(
        "SCIM provisioning",
        "Settings > Security > SCIM",
        "Enterprise",
        "the lifetime of a SCIM token",
        "1 year",
    ),
    Feature(
        "custom roles",
        "Settings > Members > Roles",
        "Business",
        "the maximum number of custom roles",
        "20",
    ),
    Feature(
        "purchase orders",
        "Expenses > Purchase orders",
        "Business",
        "the default purchase order number format",
        "PO-YYYY-NNNN",
    ),
    Feature(
        "data export",
        "Settings > Data > Export",
        "Starter",
        "the list of export formats",
        "CSV, JSON and XLSX",
    ),
    Feature(
        "scheduled reports",
        "Reports > Schedules",
        "Team",
        "the most frequent report schedule",
        "daily at 06:00 workspace time",
    ),
    Feature(
        "tax rules",
        "Settings > Taxes",
        "Starter",
        "the maximum number of tax rates per line item",
        "3",
    ),
    Feature(
        "sandbox workspaces",
        "Settings > Developer > Sandbox",
        "Enterprise",
        "the sandbox data reset interval",
        "every 30 days",
    ),
    Feature(
        "webhooks",
        "Settings > Developer > Webhooks",
        "Team",
        "the webhook retry window",
        "72 hours",
    ),
    Feature(
        "IP allowlisting",
        "Settings > Security > IP allowlist",
        "Enterprise",
        "the maximum number of allowlisted IP ranges",
        "100",
    ),
    Feature(
        "credit notes",
        "Invoices > Credit notes",
        "Starter",
        "the credit note numbering prefix",
        "CN",
    ),
    Feature(
        "retainers",
        "Invoices > Retainers",
        "Business",
        "the minimum retainer term",
        "one month",
    ),
    Feature(
        "bulk invoice import",
        "Invoices > Import",
        "Team",
        "the maximum number of rows per import file",
        "5000",
    ),
    Feature(
        "time-off tracking",
        "Time > Time off",
        "Business",
        "the accrual calculation period",
        "each pay period",
    ),
)

INTEGRATIONS: tuple[Integration, ...] = (
    Integration(
        "Slack",
        "messaging",
        "within 1 minute",
        "invoice-paid and approval-request notifications",
        "Starter",
    ),
    Integration(
        "Microsoft Teams", "messaging", "within 1 minute", "approval-request notifications", "Team"
    ),
    Integration(
        "QuickBooks Online",
        "accounting",
        "every 15 minutes",
        "invoices, payments and customers",
        "Team",
    ),
    Integration("Xero", "accounting", "every 15 minutes", "invoices, bills and contacts", "Team"),
    Integration(
        "NetSuite",
        "accounting",
        "every 60 minutes",
        "invoices, vendor bills and the chart of accounts",
        "Enterprise",
    ),
    Integration(
        "Stripe", "payments", "every 5 minutes", "payments, refunds and payouts", "Starter"
    ),
    Integration(
        "Gusto", "payroll", "once per pay run", "approved hours and reimbursements", "Business"
    ),
    Integration(
        "Google Calendar", "calendar", "every 10 minutes", "events as draft time entries", "Starter"
    ),
    Integration(
        "Outlook Calendar",
        "calendar",
        "every 10 minutes",
        "events as draft time entries",
        "Starter",
    ),
    Integration("Jira", "project management", "every 30 minutes", "issues and worklogs", "Team"),
    Integration("Asana", "project management", "every 30 minutes", "projects and tasks", "Team"),
    Integration(
        "HubSpot", "CRM", "every 20 minutes", "companies, contacts and closed deals", "Business"
    ),
    Integration(
        "Salesforce",
        "CRM",
        "every 20 minutes",
        "accounts and closed-won opportunities",
        "Enterprise",
    ),
    Integration("Shopify", "ecommerce", "every 15 minutes", "orders and refunds", "Team"),
    Integration(
        "Zapier",
        "automation",
        "on each trigger event",
        "any Tallyhive trigger or action",
        "Starter",
    ),
)

ERROR_KINDS: tuple[ErrorKind, ...] = (
    ErrorKind(
        "401",
        "{obj} authorization expired",
        "an expired {obj} authorization token",
        "to reconnect {obj} under Settings > Integrations",
    ),
    ErrorKind(
        "403",
        "{obj} permission denied",
        "a connected {obj} account without admin permission",
        "to reconnect {obj} using an account with admin rights",
    ),
    ErrorKind(
        "429",
        "{obj} rate limited",
        "too many requests to {obj} in a short period",
        "to wait 15 minutes and then retry the sync",
    ),
    ErrorKind(
        "422",
        "{obj} rejected a field",
        "a field mapping that {obj} rejects",
        "to review the field mappings under Settings > Integrations > {obj} > Mappings",
    ),
    ErrorKind(
        "409",
        "{obj} sync conflict",
        "a record edited in both Tallyhive and {obj} at the same time",
        "to choose the version to keep in the Sync conflicts queue",
    ),
)

POLICIES: tuple[Policy, ...] = (
    Policy(
        "Data residency",
        "The available data regions are the United States, the European Union and Australia.",
        "the United States, the European Union and Australia",
        (
            "Customer data is stored in the region chosen at signup.",
            "Moving a workspace to another region requires a migration request to support.",
        ),
        ("Which data regions can I choose for my workspace?", "Where can Tallyhive host my data?"),
    ),
    Policy(
        "Free trial",
        "The free trial length is 21 days.",
        "21 days",
        ("No payment card is needed to start a trial.", "Trials include every Business feature."),
        ("How long is the free trial?",),
    ),
    Policy(
        "Deleting your account",
        "The grace period after an account deletion request is 30 days.",
        "30 days",
        (
            "Owners can cancel the deletion from the confirmation email during this time.",
            "Data is purged from backups within 90 days of deletion.",
        ),
        ("How long do I have to undo deleting my account?",),
    ),
    Policy(
        "Refund policy",
        "Annual subscriptions are refundable within 30 days of purchase.",
        "within 30 days of purchase",
        ("Monthly subscriptions are not refunded for partial months.",),
        ("Can I get my money back on an annual subscription?",),
    ),
    Policy(
        "Uptime commitment",
        "The uptime commitment for paid plans is 99.9 percent per calendar month.",
        "99.9 percent per calendar month",
        ("Service credits are issued automatically when the commitment is missed.",),
        ("What uptime does Tallyhive guarantee?",),
    ),
    Policy(
        "Backups",
        "The backup frequency for workspace data is every 6 hours.",
        "every 6 hours",
        ("Backups are encrypted and stored in a second availability zone.",),
        ("How often is my workspace backed up?",),
    ),
    Policy(
        "Two-factor authentication",
        "The supported two-factor methods are authenticator apps and hardware security keys.",
        "authenticator apps and hardware security keys",
        ("SMS codes are not supported.", "Owners can require two-factor sign-in for all members."),
        ("What 2FA options can members use?",),
    ),
    Policy(
        "Password requirements",
        "The minimum password length is 12 characters.",
        "12 characters",
        ("Passwords found in known breach lists are rejected.",),
        ("How long does a password need to be?",),
    ),
    Policy(
        "Session timeout",
        "The default idle session timeout is 12 hours.",
        "12 hours",
        ("Business and Enterprise owners can shorten the timeout.",),
        ("When do idle sessions get signed out?",),
    ),
    Policy(
        "Mobile apps",
        "The mobile apps are available for iOS 16 or later and Android 11 or later.",
        "iOS 16 or later and Android 11 or later",
        ("Receipt scanning works offline and uploads when a connection returns.",),
        ("Which phone operating systems does the mobile app support?",),
    ),
    Policy(
        "Supported browsers",
        "The supported browsers are the latest two versions of Chrome, Firefox, Safari and Edge.",
        "the latest two versions of Chrome, Firefox, Safari and Edge",
        ("Internet Explorer is not supported.",),
        ("Which web browsers are supported?",),
    ),
    Policy(
        "Subscription payment methods",
        "The accepted payment methods for subscriptions are credit card and ACH bank transfer.",
        "credit card and ACH bank transfer",
        ("Enterprise contracts can also be paid by wire transfer.",),
        ("How can I pay for my Tallyhive subscription?",),
    ),
    Policy(
        "Data processing agreement",
        "The data processing agreement is signed from Settings > Legal > DPA.",
        "Settings > Legal > DPA",
        ("The agreement incorporates the EU standard contractual clauses.",),
        ("Where do I sign the DPA?",),
    ),
    Policy(
        "Sub-processors",
        "The notice period for new sub-processors is 30 days.",
        "30 days",
        ("The current sub-processor list is published in the Trust Center.",),
        ("How much notice is given before a new sub-processor is added?",),
    ),
    Policy(
        "Keyboard shortcuts",
        "The shortcut for creating a new invoice is N followed by I.",
        "N followed by I",
        ("Press the question mark key to see every shortcut.",),
        ("What is the keyboard shortcut for a new invoice?",),
    ),
    Policy(
        "Invoice numbering",
        "The default invoice number format is INV followed by a six-digit sequence.",
        "INV followed by a six-digit sequence",
        ("Numbering restarts only when an owner changes the prefix.",),
        ("What format do invoice numbers use by default?",),
    ),
)

FEATURE_CONTEXT = (
    "Workspace owners and admins can change this setting.",
    "Changes take effect immediately for every member of the workspace.",
    "Every change to this setting is recorded in the audit log.",
    "The mobile apps show the same configuration in read-only form.",
)

RELEASE_TEMPLATES = (
    "{feature} now loads faster in large workspaces.",
    "Fixed an issue where the {integration} sync could stall after a password change.",
    "{feature} gained a new filter for archived clients.",
    "The {integration} connection screen now shows the time of the last successful sync.",
    "Improved accessibility labels across {feature}.",
)

MULTI_HOP_ATTRIBUTES = ("api-rate-limit", "audit-retention", "seat-cap", "support-response")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _bare(name: str) -> str:
    """Feature name without a leading article, for titles and slugs."""
    return name.removeprefix("the ")


class _Builder:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.corpus: dict[str, Document] = {}
        self.queries: dict[str, Query] = {}
        self.qrels: Qrels = {}
        self.answers: dict[str, list[str]] = {}

    def doc(self, doc_id: str, title: str, fact: str, context: list[str]) -> str:
        """Render a document with its fact sentence at a seeded position among the context."""
        sentences = list(context)
        self.rng.shuffle(sentences)
        sentences.insert(self.rng.randrange(len(sentences) + 1), fact)
        self.corpus[doc_id] = Document(doc_id, title, " ".join(sentences))
        return doc_id

    def query(self, kind: str, text: str, answer: str, judged: dict[str, int]) -> None:
        qid = f"q{len(self.queries) + 1:03d}"
        self.queries[qid] = Query(qid, text, {"type": kind})
        self.qrels[qid] = judged
        self.answers[qid] = [answer]

    def pick(self, options: tuple[str, ...]) -> str:
        return options[self.rng.randrange(len(options))]


def build_synthetic_dataset(seed: int = DEFAULT_SEED) -> Dataset:
    """Build the SYNTHETIC Tallyhive help-center dataset deterministically from ``seed``."""
    b = _Builder(seed)

    plan_attr_docs: dict[tuple[str, str], str] = {}
    overview_docs: dict[str, str] = {}
    for i, plan in enumerate(PLANS):
        included = [f.name for f in FEATURES if PLANS.index(f.min_plan) == i][:3]
        overview_docs[plan] = b.doc(
            f"plan-{slugify(plan)}-overview",
            f"{plan} plan overview",
            f"The {plan} plan is designed for {PLAN_AUDIENCE[plan]}.",
            [
                f"Plans at this level add {', '.join(included)}.",
                "Every plan includes unlimited clients and invoices.",
            ],
        )
        for attr in PLAN_ATTRIBUTES:
            context = list(attr.context)
            if i + 1 < len(PLANS):
                context.append(f"Upgrading to {PLANS[i + 1]} changes this to {attr.values[i + 1]}.")
            plan_attr_docs[(attr.key, plan)] = b.doc(
                f"plan-{slugify(plan)}-{attr.key}",
                f"{attr.title} on the {plan} plan",
                f"{_cap(attr.subject)} on the {plan} plan is {attr.values[i]}.",
                context,
            )

    setup_docs: dict[str, str] = {}
    availability_docs: dict[str, str] = {}
    for feature in FEATURES:
        slug = slugify(_bare(feature.name))
        setup_docs[feature.name] = b.doc(
            f"feature-{slug}-setup",
            f"Setting up {_bare(feature.name)}",
            f"The setting for {feature.name} is under {feature.path}.",
            [
                f"{_cap(feature.limit_subject)} is {feature.limit_value}.",
                *b.rng.sample(FEATURE_CONTEXT, 2),
            ],
        )
        availability_docs[feature.name] = b.doc(
            f"feature-{slug}-availability",
            f"Plan availability for {_bare(feature.name)}",
            f"The minimum plan for {feature.name} is {feature.min_plan}.",
            [
                f"Workspaces on lower plans see an upgrade prompt when they open {feature.path}.",
                f"Admins can request a 14-day trial of {feature.name} from the billing page.",
            ],
        )

    connect_docs: dict[str, str] = {}
    sync_docs: dict[str, str] = {}
    for integ in INTEGRATIONS:
        slug = slugify(integ.name)
        connect_docs[integ.name] = b.doc(
            f"integration-{slug}-connect",
            f"Connecting {integ.name}",
            f"The minimum plan for the {integ.name} integration is {integ.min_plan}.",
            [
                f"The {integ.name} integration is set up under Settings > Integrations.",
                f"Connecting requires an account with admin rights in {integ.name}.",
                f"{integ.name} appears in the {integ.category} category of the marketplace.",
            ],
        )
        sync_docs[integ.name] = b.doc(
            f"integration-{slug}-sync",
            f"How the {integ.name} sync works",
            f"The {integ.name} sync interval is {integ.interval}.",
            [
                f"The objects synced with {integ.name} are {integ.objects}.",
                "A manual sync can be started from the integration page at any time.",
            ],
        )

    pairs = [(kind, integ) for kind in ERROR_KINDS for integ in INTEGRATIONS]
    errors = b.rng.sample(pairs, 30)
    error_docs: list[tuple[str, str, ErrorKind, Integration]] = []
    for n, (kind, integ) in enumerate(errors):
        code = f"TH-{kind.prefix}{n:02d}"
        doc_id = b.doc(
            f"error-{code.lower()}",
            f"Error {code}: {kind.title.format(obj=integ.name)}",
            f"Error {code} is caused by {kind.cause.format(obj=integ.name)}.",
            [
                f"The fix for error {code} is {kind.fix.format(obj=integ.name)}.",
                "If the error persists, contact support with the request ID from the banner.",
            ],
        )
        error_docs.append((doc_id, code, kind, integ))

    policy_docs: dict[str, str] = {}
    for policy in POLICIES:
        policy_docs[policy.title] = b.doc(
            f"policy-{slugify(policy.title)}", policy.title, policy.fact, list(policy.context)
        )

    for month in range(24):
        year, mon = 2024 + month // 12, month % 12 + 1
        lines = []
        for template in b.rng.sample(RELEASE_TEMPLATES, 3):
            feature_name = _cap(b.pick(tuple(_bare(f.name) for f in FEATURES)))
            integration_name = b.pick(tuple(i.name for i in INTEGRATIONS))
            lines.append(template.format(feature=feature_name, integration=integration_name))
        b.doc(
            f"release-{year}-{mon:02d}",
            f"Release notes {year}-{mon:02d}",
            lines[0],
            lines[1:],
        )

    _add_queries(
        b, plan_attr_docs, overview_docs, setup_docs, availability_docs, connect_docs, sync_docs
    )
    sampled_errors = b.rng.sample(error_docs, 10)
    fix_questions = (
        "How do I fix error {c}?",
        "I'm seeing {c} in the banner. How do I resolve it?",
    )
    for doc_id, code, kind, integ in sampled_errors[:6]:
        b.query(
            "error_fix",
            b.pick(fix_questions).format(c=code),
            kind.fix.format(obj=integ.name),
            {doc_id: 2, connect_docs[integ.name]: 1},
        )
    for doc_id, code, kind, integ in sampled_errors[6:]:
        b.query(
            "error_cause",
            f"What causes error {code}?",
            kind.cause.format(obj=integ.name),
            {doc_id: 2, connect_docs[integ.name]: 1},
        )
    for policy in b.rng.sample(POLICIES, 6):
        b.query("policy", b.pick(policy.questions), policy.answer, {policy_docs[policy.title]: 2})
    _add_multi_hop(b, plan_attr_docs, availability_docs, connect_docs)

    return Dataset("synthetic-helpcenter", b.corpus, b.queries, b.qrels, b.answers)


def _add_queries(
    b: _Builder,
    plan_attr_docs: dict[tuple[str, str], str],
    overview_docs: dict[str, str],
    setup_docs: dict[str, str],
    availability_docs: dict[str, str],
    connect_docs: dict[str, str],
    sync_docs: dict[str, str],
) -> None:
    combos = [(attr, plan) for attr in PLAN_ATTRIBUTES for plan in PLANS]
    for attr, plan in b.rng.sample(combos, 10):
        b.query(
            "plan_attribute",
            b.pick(attr.questions).format(plan=plan),
            attr.values[PLANS.index(plan)],
            {plan_attr_docs[(attr.key, plan)]: 2, overview_docs[plan]: 1},
        )

    features = b.rng.sample(FEATURES, 14)
    for feature in features[:5]:
        template = b.pick(
            ("Where do I set up {f}?", "Where is the setting for {f}?", "How do I turn on {f}?")
        )
        b.query(
            "feature_location",
            template.format(f=feature.name),
            feature.path,
            {setup_docs[feature.name]: 2, availability_docs[feature.name]: 1},
        )
    for feature in features[5:10]:
        template = b.pick(("Which plan do I need for {f}?", "What is the cheapest plan with {f}?"))
        b.query(
            "feature_plan",
            template.format(f=feature.name),
            feature.min_plan,
            {availability_docs[feature.name]: 2, setup_docs[feature.name]: 1},
        )
    for feature in features[10:14]:
        b.query(
            "feature_limit",
            f"What is {feature.limit_subject}?",
            feature.limit_value,
            {setup_docs[feature.name]: 2, availability_docs[feature.name]: 1},
        )

    integrations = b.rng.sample(INTEGRATIONS, 7)
    for integ in integrations[:5]:
        template = b.pick(
            ("How often does {i} sync with Tallyhive?", "What is the sync interval for {i}?")
        )
        b.query(
            "integration_sync",
            template.format(i=integ.name),
            integ.interval,
            {sync_docs[integ.name]: 2, connect_docs[integ.name]: 1},
        )
    for integ in integrations[5:]:
        b.query(
            "integration_objects",
            f"What data does the {integ.name} integration sync?",
            integ.objects,
            {sync_docs[integ.name]: 2, connect_docs[integ.name]: 1},
        )


def _add_multi_hop(
    b: _Builder,
    plan_attr_docs: dict[tuple[str, str], str],
    availability_docs: dict[str, str],
    connect_docs: dict[str, str],
) -> None:
    attrs = {a.key: a for a in PLAN_ATTRIBUTES}
    for feature in b.rng.sample(FEATURES, 5):
        attr = attrs[b.pick(MULTI_HOP_ATTRIBUTES)]
        b.query(
            "multi_hop",
            f"What is the {attr.short} on the cheapest plan that includes {feature.name}?",
            attr.values[PLANS.index(feature.min_plan)],
            {
                plan_attr_docs[(attr.key, feature.min_plan)]: 2,
                availability_docs[feature.name]: 1,
            },
        )
    for integ in b.rng.sample(INTEGRATIONS, 3):
        attr = attrs[b.pick(MULTI_HOP_ATTRIBUTES)]
        b.query(
            "multi_hop",
            f"What {attr.short} applies on the minimum plan for the {integ.name} integration?",
            attr.values[PLANS.index(integ.min_plan)],
            {plan_attr_docs[(attr.key, integ.min_plan)]: 2, connect_docs[integ.name]: 1},
        )
