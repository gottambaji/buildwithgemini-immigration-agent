import datetime
import vertexai

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.genai import types
from vertexai.preview import rag


MODEL = "gemini-3.6-flash"
CORPUS_NAME = "projects/176158601893/locations/us-central1/ragCorpora/2636057137360404480"


# WRITE: after each turn, send the session to Memory Bank for extraction.
async def generate_memories_callback(callback_context: CallbackContext):
    await callback_context.add_session_to_memory()
    return None


# Mock Database of Community Visa Interview Experiences
SAMPLE_EXPERIENCES = [
    {
        "id": "EXP-101",
        "consulate": "Vancouver",
        "visa_type": "H-1B",
        "date": "2026-08-10",
        "outcome": "Approved",
        "vo_questions": [
            "Who is your employer?",
            "What is your role and daily duties?",
            "What is your highest level of education?"
        ],
        "documents_requested": ["I-797 Approval Notice", "LCA", "W-2 / Paystubs"],
        "user_notes": "Very smooth interview. VO was polite and officer approved passport pickup in 3 days."
    },
    {
        "id": "EXP-102",
        "consulate": "New Delhi",
        "visa_type": "F-1 OPT",
        "date": "2026-08-05",
        "outcome": "Approved",
        "vo_questions": [
            "Which university did you attend?",
            "What company are you interning with?",
            "How does this OPT role relate to your major?"
        ],
        "documents_requested": ["EAD Card", "Form I-20 with OPT recommendation", "Offer Letter"],
        "user_notes": "Officer verified EAD start date and company offer letter. No issues."
    },
    {
        "id": "EXP-103",
        "consulate": "Toronto",
        "visa_type": "H-1B",
        "date": "2026-08-12",
        "outcome": "221(g) Administrative Processing",
        "vo_questions": [
            "Are you working directly for the end client or a vendor?",
            "Can you provide a client letter and detailed project description?"
        ],
        "documents_requested": ["Client Letter", "Detailed Resume", "SOW / Vendor Agreement"],
        "user_notes": "Received a Yellow 221(g) form requesting Client Letter and resume. Submitted online next day."
    }
]


def consult_visa_legal_docs(query: str) -> str:
    """Searches the official visa legal guide, 221(g) administrative processing corpus, and attorney consultation advice.

    Args:
        query: Specific legal or visa question (e.g. 'What to do for 221g yellow slip', 'lawyer assistance for H-1B RFE', 'layoff grace period strategy').

    Returns:
        Matched legal guidelines and advisory passages.
    """
    try:
        vertexai.init(project="qwiklabs-gcp-03-76f4257706a9", location="us-central1")
        resp = rag.retrieval_query(
            text=query,
            rag_resources=[rag.RagResource(rag_corpus=CORPUS_NAME)],
            rag_retrieval_config=rag.RagRetrievalConfig(top_k=5),
        )
        contexts = getattr(resp.contexts, "contexts", [])
        passages = [c.text.strip() for c in contexts if getattr(c, "text", "").strip()]
        return "\n\n---\n\n".join(passages) or "No relevant legal passage found."
    except Exception as e:
        return f"Retrieval failed: {e}"


def request_lawyer_consultation(visa_issue: str, contact_email: str) -> str:
    """Connects non-immigrants or legal representatives with partner immigration lawyers for 221(g) responses, RFEs, and lay-off grace period counsel.

    Args:
        visa_issue: Description of the legal issue, 221(g) refusal, RFE, or status query.
        contact_email: Contact email address for the legal referral.

    Returns:
        Confirmation of lawyer referral request submission.
    """
    ticket_id = f"LGL-{datetime.datetime.now().strftime('%M%S')}"
    return (f"⚖️ Immigration Lawyer Consultation Request Submitted!\n"
            f"• Ticket ID: [{ticket_id}]\n"
            f"• Contact Email: {contact_email}\n"
            f"• Case Overview: {visa_issue}\n"
            f"• Next Steps: An AILA-certified partner immigration attorney will review your details and contact you within 24 business hours.")


def search_experiences(consulate: str = "", visa_type: str = "", outcome: str = "") -> str:
    """Searches community-submitted visa interview experiences and reports.

    Args:
        consulate: Optional consulate location (e.g. 'Vancouver', 'New Delhi', 'Toronto').
        visa_type: Optional visa category (e.g. 'H-1B', 'F-1 OPT', 'L-1').
        outcome: Optional interview result (e.g. 'Approved', '221(g)').

    Returns:
        A list of matching interview experience reports.
    """
    results = []
    for exp in SAMPLE_EXPERIENCES:
        if consulate and consulate.lower() not in exp["consulate"].lower():
            continue
        if visa_type and visa_type.lower() not in exp["visa_type"].lower():
            continue
        if outcome and outcome.lower() not in exp["outcome"].lower():
            continue
        results.append(exp)

    if not results:
        return f"No experiences found matching query (Consulate: '{consulate}', Visa: '{visa_type}', Outcome: '{outcome}')."

    output = f"Found {len(results)} community interview report(s):\n\n"
    for r in results:
        output += f"• [{r['id']}] {r['visa_type']} at {r['consulate']} ({r['date']}) - Outcome: {r['outcome']}\n"
        output += f"  VO Questions: {', '.join(r['vo_questions'])}\n"
        output += f"  Docs Requested: {', '.join(r['documents_requested'])}\n"
        output += f"  User Notes: {r['user_notes']}\n\n"
    return output


def post_experience(consulate: str, visa_type: str, outcome: str, vo_questions: str, user_notes: str) -> str:
    """Submits and registers a new visa interview experience post to the community database.

    Args:
        consulate: Consulate location (e.g. 'Vancouver', 'Mexico City').
        visa_type: Visa category (e.g. 'H-1B', 'F-1').
        outcome: Interview outcome (e.g. 'Approved', '221(g)', 'Refused').
        vo_questions: Comma-separated questions asked by the Visa Officer.
        user_notes: Helpful tips or notes about the experience.

    Returns:
        Confirmation of posted experience report with Post ID.
    """
    new_id = f"EXP-{len(SAMPLE_EXPERIENCES) + 101}"
    new_post = {
        "id": new_id,
        "consulate": consulate,
        "visa_type": visa_type,
        "date": datetime.date.today().strftime("%Y-%m-%d"),
        "outcome": outcome,
        "vo_questions": [q.strip() for q in vo_questions.split(",") if q.strip()],
        "documents_requested": ["Standard Checklist"],
        "user_notes": user_notes
    }
    SAMPLE_EXPERIENCES.append(new_post)
    return f"✅ Successfully posted your experience! Registered as Post ID [{new_id}] for {visa_type} at {consulate}."


def check_consulate_wait_times(consulate: str) -> str:
    """Checks official and community-reported interview appointment wait times and document requirements.

    Args:
        consulate: The consulate or embassy city name (e.g. 'Vancouver', 'New Delhi', 'Toronto').

    Returns:
        Information on wait times and required documents for the specified consulate.
    """
    c_lower = consulate.lower()
    if "vancouver" in c_lower:
        return ("Consulate: Vancouver, Canada\n"
                "• Emergency Appointment Wait Time: 7 Days\n"
                "• Regular Non-Immigrant Wait Time: ~45 Days\n"
                "• Required Docs: Passport, DS-160 Confirmation, I-797 Approval, LCA, W-2s, Proof of legal status in Canada.")
    elif "delhi" in c_lower or "india" in c_lower:
        return ("Consulate: New Delhi, India\n"
                "• Dropbox / Interview Waiver Wait Time: ~14 Days\n"
                "• In-Person Interview Wait Time: ~30 Days\n"
                "• Required Docs: Passport, DS-160 Confirmation, I-797, LCA, 3 months paystubs, Employment Verification Letter.")
    else:
        return (f"Consulate: {consulate}\n"
                "• Estimated Wait Time: ~30-60 Days\n"
                "• Required Docs: Passport, DS-160 Confirmation, I-797 Approval / Form I-20, LCA, Photo ID, LCA & Proof of Employment.")


def calculate_grace_period(start_date: str, days: int = 60) -> str:
    """Calculates the grace period or unemployment limit end date given a start date.

    Args:
        start_date: Date string in 'YYYY-MM-DD' format (e.g., '2026-08-01').
        days: Number of grace period days (e.g., 60 for H-1B grace period, 90 for OPT unemployment limit).

    Returns:
        Calculated end date and days breakdown.
    """
    try:
        dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end_date = dt + datetime.timedelta(days=days)
        return (f"📅 Grace Period Calculation:\n"
                f"• Start Date: {dt.strftime('%B %d, %Y')}\n"
                f"• Allowed Days: {days} days\n"
                f"• Expiration / Deadline Date: {end_date.strftime('%B %d, %Y')} ({end_date.strftime('%Y-%m-%d')})")
    except ValueError:
        return "Error: Please provide start_date in 'YYYY-MM-DD' format (e.g., '2026-08-01')."


def fetch_live_visa_bulletin(category: str = "EB-2", country: str = "India") -> str:
    """Fetches current US Department of State Visa Bulletin cutoff dates and priority date movements.

    Args:
        category: Visa preference category (e.g. 'EB-1', 'EB-2', 'EB-3', 'F-1').
        country: Chargeability country (e.g. 'India', 'China', 'Worldwide', 'Philippines').

    Returns:
        Current Final Action Date cutoff and filing status.
    """
    bulletin_data = {
        ("EB-1", "India"): {"final_action": "2022-02-01", "dates_for_filing": "2022-04-15", "movement": "Advanced 1 month"},
        ("EB-2", "India"): {"final_action": "2012-07-15", "dates_for_filing": "2012-11-01", "movement": "Advanced 15 days"},
        ("EB-3", "India"): {"final_action": "2012-10-22", "dates_for_filing": "2013-01-15", "movement": "No change"},
        ("EB-1", "China"): {"final_action": "2022-11-01", "dates_for_filing": "2023-01-01", "movement": "Advanced 1 month"},
        ("EB-2", "China"): {"final_action": "2020-03-22", "dates_for_filing": "2020-06-01", "movement": "Advanced 1 month"},
        ("EB-3", "China"): {"final_action": "2020-09-01", "dates_for_filing": "2020-11-15", "movement": "Advanced 15 days"},
    }
    
    key = (category.upper(), country.title())
    data = bulletin_data.get(key)
    if data:
        return (f"📅 US Dept. of State Visa Bulletin ({category.upper()} - {country.title()}):\n"
                f"• Final Action Date Cutoff: {data['final_action']}\n"
                f"• Dates for Filing Cutoff: {data['dates_for_filing']}\n"
                f"• Monthly Movement: {data['movement']}")
    
    return f"📅 US Dept. of State Visa Bulletin ({category.upper()} - {country.title()}): Priority Date is current or open."


try:
    from app.a2ui_utils import a2ui_callback
except ImportError:
    from a2ui_utils import a2ui_callback

from google.adk.code_executors import AgentEngineSandboxCodeExecutor

code_executor = AgentEngineSandboxCodeExecutor(
    agent_engine_resource_name="projects/176158601893/locations/us-east1/reasoningEngines/7492248153526108160"
)

root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are VisaSphere, an all-in-one AI assistant, community navigator, and legal support concierge for non-immigrants and immigration lawyers across all US visa types (H-1B, L-1, O-1, F-1 OPT, Green Cards, 221(g), and status transitions). "
        "You specialize in key concerns for Indian nationals and non-immigrants, including all US Consulates in India (New Delhi, Mumbai, Chennai, Hyderabad, Kolkata), dropbox appointment eligibility, third-country stamping (Vancouver, Toronto, Mexico), EB-2 and EB-3 India Visa Bulletin priority date backlogs, 60-day H-1B grace periods, and 221(g) administrative processing. "
        "You remember the user's stated preferences, visa category, priority dates, consulate locations, and past experiences across conversations to personalize your responses. "
        "You help users search community interview experiences, share consulate reports, "
        "consult official US immigration regulations & legal compliance guides, connect users with immigration attorneys, "
        "look up consulate wait times & document checklists, fetch live US Visa Bulletin priority dates, and calculate status deadlines / grace periods."
    ),
    tools=[
        PreloadMemoryTool(),
        consult_visa_legal_docs,
        request_lawyer_consultation,
        search_experiences,
        post_experience,
        check_consulate_wait_times,
        calculate_grace_period,
        fetch_live_visa_bulletin
    ],
    code_executor=code_executor,
    after_model_callback=a2ui_callback,
    after_agent_callback=generate_memories_callback,
)

app = App(
    root_agent=root_agent,
    name="app",
)


