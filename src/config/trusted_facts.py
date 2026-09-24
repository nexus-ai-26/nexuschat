"""Trusted, application-owned programme and community facts.

This module is the single source of truth for organizer identities.  Chat
messages and model memory must not be used to fill in missing contacts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TrustedApplicationFact:
    name: str
    contact: str | None
    email: str | None
    community_role: str
    programme_role: str
    responsibilities: tuple[str, ...]
    when_to_contact: str
    aliases: tuple[str, ...] = ()


TRUSTED_APPLICATION_FACTS: tuple[TrustedApplicationFact, ...] = (
    TrustedApplicationFact(
        name="Diane",
        contact="+250783188655",
        email="unipods.regional@undp.org",
        community_role="community/group admin",
        programme_role="Programme Facilitator / Organizer",
        responsibilities=(
            "team declarations and bot deployment",
            "programme facilitation / organizer matters",
        ),
        when_to_contact="team declarations, bot deployment, or organizer matters",
        aliases=("diane", "diana"),
    ),
    TrustedApplicationFact(
        name="Gift NTULI",
        contact="+263774094822",
        email=None,
        community_role="community/group admin",
        programme_role="Programme Facilitator / Organizer",
        responsibilities=("Microsoft session leader",),
        when_to_contact="Microsoft session questions",
        aliases=("gift", "ntuli"),
    ),
    TrustedApplicationFact(
        name="Munira UNDP",
        contact="+250786387244",
        email=None,
        community_role="community/group admin",
        programme_role="Programme Facilitator / Organizer",
        responsibilities=(
            "MIT Course-related issues, including invitation-link/course support",
        ),
        when_to_contact="MIT Course, invitation-link, or course-support issues",
        aliases=("munira", "undp"),
    ),
    TrustedApplicationFact(
        name="Jeovaire Umukundwa",
        contact=None,
        email=None,
        community_role="community/group admin",
        programme_role="Programme Facilitator / Organizer",
        responsibilities=("Microsoft Session leader (secondary)",),
        when_to_contact="Microsoft session questions, especially secondary support",
        aliases=("jeovaire", "umukundwa"),
    ),
)


_ADMIN_QUERY_RE = re.compile(
    r"\b(?:admin(?:istrator)?s?|organizers?|facilitator(?:s)?|contact|phone|"
    r"number|whatsapp|handles?|responsib(?:le|ility)|who\s+should\s+i)\b",
    re.IGNORECASE,
)
_MIT_RE = re.compile(r"\b(?:mit|course|invitation|invite)\b", re.IGNORECASE)
_MICROSOFT_RE = re.compile(
    r"\b(?:microsoft|session\s+leader|session\s+2)\b", re.IGNORECASE
)
_DIANE_RE = re.compile(
    r"\b(?:team\s+declaration|team\s+declarations|deployment|deploy(?:ing|ment)?)\b",
    re.IGNORECASE,
)
_CONTACT_REQUEST_RE = re.compile(
    r"\b(?:phone|number|contact|call|whatsapp)\b", re.IGNORECASE
)


def relevant_trusted_application_facts(
    query: str | None,
) -> tuple[TrustedApplicationFact, ...]:
    """Return only trusted organizer facts relevant to a user query."""

    text = (query or "").casefold()
    if not text:
        return ()

    mentioned_names = {
        fact.name
        for fact in TRUSTED_APPLICATION_FACTS
        if any(re.search(rf"\b{re.escape(alias)}\b", text) for alias in fact.aliases)
    }
    if mentioned_names:
        names = mentioned_names
    elif _DIANE_RE.search(text):
        names = {"Diane"}
    elif _MIT_RE.search(text):
        names = {"Munira UNDP"}
    elif _MICROSOFT_RE.search(text):
        names = {"Gift NTULI", "Jeovaire Umukundwa"}
    elif _ADMIN_QUERY_RE.search(text):
        names = {fact.name for fact in TRUSTED_APPLICATION_FACTS}
    else:
        return ()

    return tuple(fact for fact in TRUSTED_APPLICATION_FACTS if fact.name in names)


def format_trusted_application_facts(
    query: str | None,
) -> str:
    """Format relevant trusted facts for the model without exposing extra contacts."""

    facts = relevant_trusted_application_facts(query)
    if not facts:
        return "No relevant trusted application facts are available."

    include_contacts = bool(_CONTACT_REQUEST_RE.search(query or ""))
    lines: list[str] = []
    for fact in facts:
        line = (
            f"- {fact.name}: {fact.community_role}; {fact.programme_role}; "
            f"responsibilities: {', '.join(fact.responsibilities)}; "
            f"when to contact: {fact.when_to_contact}."
        )
        if include_contacts and fact.contact:
            line += f" Trusted contact: {fact.contact}."
        if include_contacts and fact.email:
            line += f" Trusted email: {fact.email}."
        lines.append(line)
    return "\n".join(lines)


def trusted_contacts_for_query(query: str | None) -> tuple[str, ...]:
    """Return contacts that may survive visible-reply cleanup for this query."""

    if not _CONTACT_REQUEST_RE.search(query or ""):
        return ()
    return tuple(
        fact.contact
        for fact in relevant_trusted_application_facts(query)
        if fact.contact
    )


def trusted_admin_jids() -> set[str]:
    """Return normalized user JIDs only for directory entries with a contact."""

    return {
        f"{fact.contact.lstrip('+')}@s.whatsapp.net"
        for fact in TRUSTED_APPLICATION_FACTS
        if fact.contact
    }


def trusted_admin_display_names() -> dict[str, str]:
    """Map normalized trusted admin JIDs to safe display names for context."""

    return {
        jid: fact.name
        for jid, fact in (
            (f"{fact.contact.lstrip('+')}@s.whatsapp.net", fact)
            for fact in TRUSTED_APPLICATION_FACTS
            if fact.contact
        )
    }
