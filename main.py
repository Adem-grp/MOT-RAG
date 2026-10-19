"""Desktop chat interface for the existing baseline RAG pipeline.

Run with:
    python main.py

The RAG implementation remains in Ask.py. This file only provides the UI and
calls Ask.retrieve() and Ask.generate() for each submitted question.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk
from typing import Any

from Ask import generate, retrieve


class RagChatApp:
    """A small ChatGPT-style desktop interface for the MOT paper collection."""

    BG = "#f7f7f8"
    SIDEBAR = "#202123"
    TEXT = "#202123"
    MUTED = "#6b7280"
    ACCENT = "#10a37f"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MOT Research Assistant")
        self.root.geometry("1000x720")
        self.root.minsize(700, 500)
        self.root.configure(bg=self.BG)

        self.question_entry: ttk.Entry
        self.send_button: ttk.Button
        self.status_label: ttk.Label
        self.chat: scrolledtext.ScrolledText
        self._build_layout()
        self._show_welcome()

    def _build_layout(self) -> None:
        sidebar = tk.Frame(self.root, bg=self.SIDEBAR, width=210)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar,
            text="MOT\nResearch Assistant",
            bg=self.SIDEBAR,
            fg="white",
            font=("Segoe UI", 17, "bold"),
            justify=tk.LEFT,
            anchor="w",
            padx=20,
            pady=24,
        ).pack(fill=tk.X)

        new_chat = tk.Button(
            sidebar,
            text="+  New chat",
            command=self._new_chat,
            bg="#343541",
            fg="white",
            activebackground="#454751",
            activeforeground="white",
            relief=tk.FLAT,
            bd=0,
            font=("Segoe UI", 10),
            padx=12,
            pady=10,
            cursor="hand2",
        )
        new_chat.pack(fill=tk.X, padx=12, pady=(8, 20))

        tk.Label(
            sidebar,
            text="Answers are grounded in\nthe indexed MOT papers.",
            bg=self.SIDEBAR,
            fg="#b8b8c0",
            font=("Segoe UI", 9),
            justify=tk.LEFT,
            padx=20,
        ).pack(side=tk.BOTTOM, fill=tk.X, pady=20)

        content = tk.Frame(self.root, bg=self.BG)
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        header = tk.Frame(content, bg=self.BG, height=58)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header,
            text="MOT paper chat",
            bg=self.BG,
            fg=self.TEXT,
            font=("Segoe UI", 15, "bold"),
            padx=28,
        ).pack(side=tk.LEFT, fill=tk.Y, pady=14)
        self.status_label = ttk.Label(header, text="Ready", foreground=self.MUTED)
        self.status_label.pack(side=tk.RIGHT, padx=28)

        self.chat = scrolledtext.ScrolledText(
            content,
            wrap=tk.WORD,
            bg=self.BG,
            fg=self.TEXT,
            relief=tk.FLAT,
            borderwidth=0,
            padx=28,
            pady=18,
            font=("Segoe UI", 11),
            state=tk.DISABLED,
            cursor="arrow",
        )
        self.chat.pack(fill=tk.BOTH, expand=True)
        self.chat.tag_configure("user_name", foreground="#2563eb", font=("Segoe UI", 10, "bold"))
        self.chat.tag_configure("assistant_name", foreground=self.ACCENT, font=("Segoe UI", 10, "bold"))
        self.chat.tag_configure("message", lmargin1=18, lmargin2=18, spacing3=18)
        self.chat.tag_configure("sources", foreground=self.MUTED, font=("Segoe UI", 9), lmargin1=18, lmargin2=18)

        composer = tk.Frame(content, bg=self.BG, padx=24, pady=18)
        composer.pack(fill=tk.X)
        self.question_entry = ttk.Entry(composer, font=("Segoe UI", 11))
        self.question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=9)
        self.question_entry.bind("<Return>", self._submit_from_event)
        self.send_button = ttk.Button(composer, text="Send", command=self._submit)
        self.send_button.pack(side=tk.RIGHT, padx=(10, 0), ipadx=12, ipady=5)
        tk.Label(
            content,
            text="Press Enter to send. Responses use only the indexed paper collection.",
            bg=self.BG,
            fg=self.MUTED,
            font=("Segoe UI", 9),
        ).pack(pady=(0, 12))

    def _show_welcome(self) -> None:
        self._append_message(
            "assistant",
            "Hello! Ask me about the multi-object tracking papers in your collection. "
            "For example: “How does ByteTrack handle low-confidence detections?”",
        )

    def _append_message(self, speaker: str, text: str, sources: list[dict[str, Any]] | None = None) -> None:
        self.chat.configure(state=tk.NORMAL)
        if speaker == "user":
            self.chat.insert(tk.END, "You\n", "user_name")
        else:
            self.chat.insert(tk.END, "MOT Research Assistant\n", "assistant_name")
        self.chat.insert(tk.END, f"{text}\n", "message")
        if sources:
            source_text = "Sources: " + ", ".join(
                f"{source['source']} p.{source['page']}" for source in sources
            )
            self.chat.insert(tk.END, f"{source_text}\n", "sources")
        self.chat.insert(tk.END, "\n", "message")
        self.chat.configure(state=tk.DISABLED)
        self.chat.see(tk.END)

    def _new_chat(self) -> None:
        self.chat.configure(state=tk.NORMAL)
        self.chat.delete("1.0", tk.END)
        self.chat.configure(state=tk.DISABLED)
        self._show_welcome()
        self.question_entry.focus_set()

    def _submit_from_event(self, _event: object) -> str:
        self._submit()
        return "break"

    def _submit(self) -> None:
        question = self.question_entry.get().strip()
        if not question or str(self.send_button["state"]) == "disabled":
            return

        self.question_entry.delete(0, tk.END)
        self._append_message("user", question)
        self.send_button.configure(state=tk.DISABLED)
        self.question_entry.configure(state=tk.DISABLED)
        self.status_label.configure(text="Thinking...")
        threading.Thread(target=self._get_answer, args=(question,), daemon=True).start()

    def _get_answer(self, question: str) -> None:
        try:
            chunks = retrieve(question)
            answer = generate(question, chunks)
            self.root.after(0, self._show_answer, answer, chunks)
        except Exception as error:
            self.root.after(0, self._show_error, error)

    def _show_answer(self, answer: str, chunks: list[dict[str, Any]]) -> None:
        self._append_message("assistant", answer, chunks)
        self._finish_request()

    def _show_error(self, error: Exception) -> None:
        self._append_message(
            "assistant",
            "I could not answer that request. Make sure Ollama is running and that "
            "the paper database has been created with Ingest.py.\n\n"
            f"Details: {error}",
        )
        self._finish_request()

    def _finish_request(self) -> None:
        self.send_button.configure(state=tk.NORMAL)
        self.question_entry.configure(state=tk.NORMAL)
        self.status_label.configure(text="Ready")
        self.question_entry.focus_set()


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("vista")
    except tk.TclError:
        pass
    RagChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
