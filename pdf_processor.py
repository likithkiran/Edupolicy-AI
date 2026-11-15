"""
pdf_processor.py
Extracts text from PDF files using PyMuPDF.
Preserves document name and original page numbers for citations.
Also provides sample academic policy handbook generation for out-of-the-box testing.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import pymupdf as fitz
except ImportError:
    import fitz


@dataclass
class PageChunk:
    """Represents a single page's text with citation metadata."""
    doc_name: str        # Original PDF filename (without extension)
    page_number: int     # 1-based original page number
    text: str            # Extracted text content
    char_count: int = field(init=False)
    word_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)
        self.word_count = len(self.text.split())

    @property
    def citation(self) -> str:
        return f"{self.doc_name}, p. {self.page_number}"

    @property
    def preview(self) -> str:
        snippet = self.text.replace("\n", " ").strip()
        return snippet[:240] + ("…" if len(snippet) > 240 else "")


def extract_chunks_from_pdf(pdf_path: str, min_chars: int = 40) -> List[PageChunk]:
    """
    Open a PDF and return one PageChunk per page (skipping near-empty pages).

    Args:
        pdf_path:  Absolute or relative path to the PDF file.
        min_chars: Pages with fewer characters than this are skipped.

    Returns:
        List of PageChunk objects ordered by page number.
    """
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    chunks: List[PageChunk] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Cannot open '{pdf_path}': {e}")

    with doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            text = page.get_text("text").strip()

            if len(text) < min_chars:
                continue

            chunks.append(
                PageChunk(
                    doc_name=doc_name,
                    page_number=page_index + 1,
                    text=text,
                )
            )

    return chunks


def load_pdfs_from_folder(folder_path: str) -> List[PageChunk]:
    """
    Recursively scan a folder and extract chunks from every PDF found.

    Returns:
        Flat list of PageChunk objects across all documents.
    """
    all_chunks: List[PageChunk] = []

    if not os.path.isdir(folder_path):
        raise ValueError(f"'{folder_path}' is not a valid directory.")

    pdf_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(folder_path)
        for f in files
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{folder_path}'.")

    for pdf_path in sorted(pdf_files):
        try:
            chunks = extract_chunks_from_pdf(pdf_path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[WARNING] Skipping '{pdf_path}': {e}")

    return all_chunks


def generate_sample_policy_pdf(output_path: str = "University_Academic_Policy_Handbook.pdf") -> str:
    """
    Generates a realistic, multi-page academic regulations PDF using PyMuPDF
    so users can immediately test EduPolicy AI without uploading external documents.
    """
    doc = fitz.open()

    pages_content = [
        (
            "SECTION 1: ACADEMIC STANDING & PROBATION",
            [
                "1.1 Good Academic Standing: An undergraduate student must maintain a minimum cumulative Grade Point Average (CGPA) of 2.00 on a 4.00 scale to remain in Good Academic Standing. Graduate students require a minimum CGPA of 3.00.",
                "1.2 Academic Warning: A student whose semester GPA falls below 2.00, while maintaining a CGPA of 2.00 or higher, will receive an official Academic Warning. This serves as an advisory notice and does not appear on permanent transcripts.",
                "1.3 Academic Probation: If a student's cumulative GPA drops below 2.00 at the conclusion of any semester, the student is placed on Academic Probation. During probation, course registration is capped at a maximum of 13 credit hours per term.",
                "1.4 Removal from Probation: A student is restored to Good Academic Standing once their cumulative GPA reaches 2.00 or higher at the end of a regular semester.",
                "1.5 Academic Suspension & Dismissal: Failure to achieve a cumulative GPA of 2.00 after two consecutive semesters on Academic Probation will lead to Academic Suspension for one academic calendar year. Dismissed students have the right to submit a formal appeal within 14 calendar days of notification."
            ]
        ),
        (
            "SECTION 2: ATTENDANCE & LEAVES OF ABSENCE",
            [
                "2.1 Mandatory Attendance: Regular attendance in all lectures, tutorials, and laboratory sessions is required. Students are expected to attend at least 85% of all scheduled contact hours per course.",
                "2.2 Excessive Absence & Automatic Failure (FA): If unexcused absences exceed 15% of total course contact hours, the instructor has the statutory authority to award an automatic grade of 'FA' (Failure due to Absences).",
                "2.3 Excused Absences: Valid excuses include verified medical hospitalization, participation in university-sanctioned athletic or academic events, and bereavement in the immediate family. Proof must be submitted to the Dean of Students within 5 business days of returning to campus.",
                "2.4 Leave of Absence (LOA): A student in good standing may request a voluntary Leave of Absence for up to two consecutive regular semesters. Requests must be filed with the Registrar's Office before the end of Week 4 of the semester.",
                "2.5 Medical Withdrawal: In situations involving chronic illness or documented psychiatric emergencies, a retroactive medical withdrawal for a whole term may be granted upon approval by the University Health Committee."
            ]
        ),
        (
            "SECTION 3: GRADING SCALE, GPA & DEAN'S LIST",
            [
                "3.1 Standard Grading Scale: Grades and grade point equivalencies are assigned as follows: A (93-100%, 4.00), A- (90-92%, 3.70), B+ (87-89%, 3.30), B (83-86%, 3.00), B- (80-82%, 2.70), C+ (77-79%, 2.30), C (73-76%, 2.00), D (60-72%, 1.00), F (<60%, 0.00).",
                "3.2 Incomplete Grade ('I'): An 'I' grade may be given only when a student has completed at least 75% of coursework with passing grades but is unable to finish due to verified unforeseen extenuating circumstances.",
                "3.3 Deadline for Incomplete Work: All missing coursework for an 'I' grade must be completed within 6 weeks into the subsequent semester. Unresolved 'I' grades convert automatically into a permanent grade of 'F'.",
                "3.4 Dean's Honor Roll: Undergraduate students who complete at least 15 graded credit hours in a regular semester with a semester GPA of 3.75 or higher and zero grades below 'B' are named to the Dean's Honor List.",
                "3.5 Grade Replacement Policy: A student may repeat up to three courses where grades of 'D' or 'F' were earned. Upon repetition, only the higher grade is calculated in the cumulative GPA, though the prior attempt remains on the transcript."
            ]
        ),
        (
            "SECTION 4: COURSE REGISTRATION & DROP/ADD DEADLINES",
            [
                "4.1 Course Add/Drop Period: The official Course Add/Drop period runs through the first seven (7) calendar days of each regular semester. Courses dropped during this window are completely expunged from the academic record with zero financial liability.",
                "4.2 Course Withdrawal Period ('W' Grade): Students may withdraw from a course between Week 2 and the close of Week 10 with an assigned grade of 'W'. The 'W' grade carries no grade point impact and does not affect the cumulative GPA.",
                "4.3 Maximum Withdrawal Allowance: An undergraduate student is permitted a maximum of four (4) course withdrawals ('W') throughout their entire undergraduate degree program.",
                "4.4 Overload Permission: Enrolling in more than 18 credit hours requires a minimum cumulative GPA of 3.50 and formal written authorization from the Department Chair and Academic Dean.",
                "4.5 Prerequisites and Corequisites: Students may not register for advanced courses without satisfying all prerequisites with a grade of 'C' or higher unless a formal Prerequisite Waiver is signed by the Academic Advisor."
            ]
        ),
        (
            "SECTION 5: ACADEMIC INTEGRITY & GENERATIVE AI POLICY",
            [
                "5.1 Academic Honor Code: The university expects honesty, fairness, and accountability. Any form of cheating, fabrication, unauthorized collaboration, or plagiarism is strictly prohibited.",
                "5.2 Plagiarism Defined: Submitting another person's words, data, code, or ideas without full and explicit attribution constitutes plagiarism. This includes self-plagiarism (submitting work previously evaluated in another class).",
                "5.3 Generative AI Regulations: The use of generative AI tools (such as ChatGPT, Claude, Gemini, or GitHub Copilot) is prohibited unless explicitly allowed by the instructor in the written course syllabus.",
                "5.4 Permitted AI Attribution: In courses where AI tools are permitted, students must include an 'AI Acknowledgement Statement' detailing the tool used, the specific prompts entered, and how the output was modified.",
                "5.5 Disciplinary Sanctions: First violations incur a zero ('0') grade on the assignment and a formal letter in the student file. A second infraction leads to an automatic course grade of 'XF' (Disciplinary Failure) and referral to the Academic Judiciary Committee for potential expulsion."
            ]
        ),
        (
            "SECTION 6: LATE SUBMISSION & GRADE APPEALS",
            [
                "6.1 Standard Late Penalty: Assignments submitted after the specified deadline without an approved extension incur a penalty of 5% deduction per 24 hours of delay, up to a maximum of five (5) days (25% total deduction).",
                "6.2 Absolute Cutoff: Work submitted more than 120 hours (5 days) past the deadline receives an automatic zero ('0') unless documented medical circumstances are verified by University Health Services.",
                "6.3 Missed Midterm and Final Examinations: A student who misses an examination due to verified illness must notify the course instructor within 48 hours of the scheduled exam and supply medical documentation.",
                "6.4 Makeup Examinations: Approved makeup examinations must be administered within ten (10) university working days of the original exam date.",
                "6.5 Grade Appeal Protocol: Students wishing to contest a final course grade must first confer informally with the instructor within 14 calendar days of grade publication. If unresolved, a formal written petition may be submitted to the Academic Appeals Committee within 30 days."
            ]
        ),
        (
            "SECTION 7: GRADUATION REQUIREMENTS & DEGREE HONORS",
            [
                "7.1 Minimum Credit Requirements: Conferral of a Bachelor of Science or Bachelor of Arts degree requires completion of a minimum of 120 approved semester credit hours, including all General Education and Major core curricula.",
                "7.2 Minimum Residency Requirement: At least 45 of the final 60 credit hours of study must be taken in direct residence at this university.",
                "7.3 Cumulative GPA for Graduation: Students must graduate with an overall cumulative GPA of at least 2.00, as well as a minimum 2.00 GPA across all courses counted toward the primary major.",
                "7.4 Latin Honors Thresholds: Degree honors are awarded based on final cumulative GPA: Cum Laude (3.60 - 3.74 CGPA), Magna Cum Laude (3.75 - 3.89 CGPA), and Summa Cum Laude (3.90 - 4.00 CGPA).",
                "7.5 Degree Clearance & Financial Audit: All students must clear library obligations, laboratory accounts, and complete exit administrative interviews prior to diploma issuance."
            ]
        )
    ]

    for title, items in pages_content:
        page = doc.new_page(width=595, height=842)  # Standard A4 size

        # Header bar
        rect_header = fitz.Rect(40, 36, 555, 76)
        page.draw_rect(rect_header, color=None, fill=(0.10, 0.22, 0.38))  # Navy blue
        page.insert_text(
            (55, 62),
            "OFFICIAL ACADEMIC REGULATIONS & POLICIES HANDBOOK",
            fontsize=12,
            color=(1.0, 1.0, 1.0),
            fontname="helv"
        )

        # Section Title
        page.insert_text(
            (50, 115),
            title,
            fontsize=14,
            color=(0.10, 0.22, 0.38),
            fontname="helv"
        )
        page.draw_line((50, 125), (545, 125), color=(0.75, 0.80, 0.88), width=1.5)

        # Content items
        y_pos = 150
        for item in items:
            # Wrap text inside bounding box
            text_rect = fitz.Rect(50, y_pos, 545, y_pos + 85)
            rc = page.insert_textbox(
                text_rect,
                item,
                fontsize=9.5,
                color=(0.15, 0.15, 0.18),
                fontname="helv",
                lineheight=1.35
            )
            y_pos += 95

        # Footer
        page.draw_line((50, 785), (545, 785), color=(0.85, 0.85, 0.85), width=0.8)
        page.insert_text(
            (50, 802),
            "Office of the Provost & Academic Affairs · Published 2024",
            fontsize=8,
            color=(0.5, 0.5, 0.5),
            fontname="helv"
        )
        page.insert_text(
            (500, 802),
            f"Page {doc.page_count}",
            fontsize=8,
            color=(0.5, 0.5, 0.5),
            fontname="helv"
        )

    doc.save(output_path)
    doc.close()
    return output_path
