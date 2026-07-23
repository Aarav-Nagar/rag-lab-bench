from __future__ import annotations

from .models import QueryFixture, SourceDocument


SAMPLE_DOCUMENTS: tuple[SourceDocument, ...] = (
    SourceDocument(
        doc_id="campus-ai-policy",
        title="Campus AI Policy",
        tags=("policy", "education"),
        text=(
            "Students may use generative AI for brainstorming, outline planning, "
            "and grammar review when the instructor permits it. Final submissions "
            "must identify meaningful AI assistance and preserve the student's own "
            "analysis, citations, and problem solving. Faculty may request process "
            "notes, prompt logs, or oral checks when authorship is unclear."
        ),
    ),
    SourceDocument(
        doc_id="library-retrieval-notes",
        title="Library Retrieval Notes",
        tags=("retrieval", "library"),
        text=(
            "The library search team ranks passages by topical match, source freshness, "
            "and citation quality. Short chunks improve precision for narrow questions, "
            "while longer chunks preserve surrounding definitions and caveats. Search "
            "results should expose document titles, passage identifiers, and quoted "
            "evidence so reviewers can trace each answer back to the collection."
        ),
    ),
    SourceDocument(
        doc_id="student-advising-faq",
        title="Student Advising FAQ",
        tags=("faq", "advising"),
        text=(
            "Advisors recommend that students check prerequisite chains before enrolling "
            "in advanced courses. If a course is full, students should join the waitlist, "
            "message the department with graduation constraints, and prepare a backup "
            "schedule. Program petitions require a concise explanation, transcript "
            "context, and any supporting instructor notes."
        ),
    ),
)


SAMPLE_QUERY_FIXTURES: tuple[QueryFixture, ...] = (
    QueryFixture(
        query_id="ai-disclosure",
        question="What should final submissions disclose when students use AI?",
        expected_doc_ids=("campus-ai-policy",),
    ),
    QueryFixture(
        query_id="retrieval-evidence",
        question="Which search results should expose passage identifiers and quoted evidence?",
        expected_doc_ids=("library-retrieval-notes",),
    ),
    QueryFixture(
        query_id="full-course-backup",
        question="What should students do when a course is full and graduation is constrained?",
        expected_doc_ids=("student-advising-faq",),
    ),
)


def load_sample_documents() -> list[SourceDocument]:
    return list(SAMPLE_DOCUMENTS)


def load_sample_query_fixtures() -> list[QueryFixture]:
    return list(SAMPLE_QUERY_FIXTURES)
