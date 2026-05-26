# =============================================================================
# main.py — ENTRY POINT
# =============================================================================
# PURPOSE: Provide a simple command-line interface to run the RAG system.
#
# HOW TO RUN:
#
#   1. Setup (first time only):
#      pip install -r requirements.txt
#      export ANTHROPIC_API_KEY="sk-ant-..."
#
#   2. Index your documents:
#      python main.py --index
#
#   3. Ask a question:
#      python main.py --query "What is the main topic of the documents?"
#
#   4. Interactive mode (index + then ask multiple questions):
#      python main.py --interactive
#
# =============================================================================

import argparse  # Built-in Python module for parsing command-line arguments
import sys
import logging
from pipeline import RAGPipeline

# Set up the root logger — this affects all loggers in all files
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s"
)

# Suppress noisy logs from third-party libraries we don't need to see
# These libraries log way too much — we just want our own logs
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def print_result(result: dict) -> None:
    """Pretty-print a query result to the terminal."""
    print("\n" + "=" * 60)
    print(f"QUESTION: {result['question']}")
    print("=" * 60)
    print(f"\nANSWER:\n{result['answer']}")

    if result.get("sources"):
        print("\n--- SOURCES USED ---")
        for i, source in enumerate(result["sources"], 1):
            print(f"\n[{i}] {source['source_file']} (score: {source['similarity_score']})")
            print(f"    Preview: {source['text_preview']}")
    print("=" * 60)


def run_interactive(pipeline: RAGPipeline) -> None:
    """
    Run an interactive question-answering loop.
    The user can type questions one by one.
    Type 'exit' or 'quit' to stop.
    """
    print("\n" + "=" * 60)
    print(" RAG PROTOTYPE — Interactive Mode")
    print(" Type your question and press Enter.")
    print(" Type 'exit' to quit.")
    print("=" * 60)

    # Infinite loop — keeps asking for questions until user exits
    while True:
        try:
            # input() prints the prompt and waits for user to type + press Enter
            question = input("\n🔍 Your question: ").strip()

            # .strip() removes leading/trailing whitespace
            # Handle empty input
            if not question:
                print("Please type a question.")
                continue

            # Check for exit commands (case-insensitive)
            if question.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            # Run the query and print results
            result = pipeline.query(question)
            print_result(result)

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully instead of showing a scary traceback
            print("\n\nInterrupted. Goodbye!")
            break


def main():
    """
    Parse command-line arguments and run the appropriate mode.
    
    argparse makes it easy to build a CLI:
    - We define expected arguments
    - argparse parses sys.argv and gives us a namespace object
    - We check which flags were set and run the right code
    """
    # ArgumentParser handles --help, --index, --query, etc.
    parser = argparse.ArgumentParser(
        description="RAG Prototype — Retrieval Augmented Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --index
  python main.py --query "What are the main topics covered?"
  python main.py --interactive
  python main.py --index --interactive   (index then go interactive)
        """
    )

    # Define the available command-line flags
    # store_true means "if flag is present, set to True; otherwise False"
    parser.add_argument(
        "--index",
        action="store_true",
        help="Index the documents in the documents/ folder"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Ask a single question and exit"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive question-answering mode"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve per query (default: 3)"
    )

    # Parse the arguments from the command line
    args = parser.parse_args()

    # If no arguments given, show help and exit
    if not any([args.index, args.query, args.interactive]):
        parser.print_help()
        sys.exit(0)

    # Initialize the pipeline (loads the embedding model)
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline()

    # Run indexing if requested
    if args.index:
        pipeline.index()

    # Run a single query if provided
    if args.query:
        result = pipeline.query(args.query, top_k=args.top_k)
        print_result(result)

    # Start interactive mode if requested
    if args.interactive:
        run_interactive(pipeline)


# This block runs only when the script is executed directly
# NOT when it's imported as a module
# It's Python's way of saying "this is the entry point"
if __name__ == "__main__":
    main()
