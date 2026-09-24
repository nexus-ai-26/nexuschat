from config import TRUSTED_APPLICATION_FACTS
from config.trusted_facts import (
    format_trusted_application_facts,
    relevant_trusted_application_facts,
)


def test_mit_question_selects_only_the_relevant_trusted_organizer():
    context = format_trusted_application_facts("Who handles the MIT course?")

    assert "Munira UNDP" in context
    assert "Diane" not in context
    assert "+250786387244" not in context


def test_explicit_contact_request_uses_only_supplied_contact():
    context = format_trusted_application_facts("What is Jeovaire's phone number?")

    assert "Jeovaire Umukundwa" in context
    assert "+" not in context
    assert "Munira UNDP" not in context


def test_trusted_directory_has_no_fabricated_jeovaire_contact():
    jeovaire = next(
        fact for fact in TRUSTED_APPLICATION_FACTS if fact.name == "Jeovaire Umukundwa"
    )

    assert jeovaire.contact is None


def test_generic_organizer_question_includes_roles_without_contacts():
    facts = relevant_trusted_application_facts("Who is the organizer?")
    context = format_trusted_application_facts("Who is the organizer?")

    assert {fact.name for fact in facts} == {
        "Diane",
        "Gift NTULI",
        "Munira UNDP",
        "Jeovaire Umukundwa",
    }
    assert "Trusted contact:" not in context


def test_team_declarations_and_deployment_are_dianes_authority():
    context = format_trusted_application_facts(
        "Who do I contact for team declarations and deployment?"
    )

    assert "Diane" in context
    assert "+250783188655" in context
    assert "unipods.regional@undp.org" in context
    assert "Gift NTULI" not in context


def test_microsoft_leaders_include_primary_and_secondary_without_invented_contacts():
    context = format_trusted_application_facts("Who leads Microsoft?")

    assert "Gift NTULI" in context
    assert "Jeovaire Umukundwa" in context
    assert "+" not in context
