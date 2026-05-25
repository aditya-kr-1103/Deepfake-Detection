from __future__ import annotations

from textwrap import dedent


def main() -> None:
    print(
        dedent(
            """
            Beyond the Pixel

            Backend:
              cd backend
              ..\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
              ..\\.venv\\Scripts\\python.exe -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000

            Frontend:
              cd frontend
              npm install
              npm run dev

            Model checkpoint:
              backend\\weights\\dfdc_effnet_b7.pt

            The PDF describes the model architecture and pipeline, but the trained
            DFDC checkpoint must be supplied separately.
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
