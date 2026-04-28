# main.py
#
# Point d'entrée CLI du système RAG → HTML prototype.
#
# Usage :
#   python main.py --pdf path/to/cahier_des_charges.pdf
#   python main.py --pdf path/to/cahier_des_charges.pdf --output prototype.html

import argparse
from pathlib import Path

from graph.workflow import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Génère un prototype HTML interactif à partir d'un cahier des charges PDF."
    )
    parser.add_argument(
        "--pdf", required=True,
        help="Chemin vers le PDF du cahier des charges"
    )
    parser.add_argument(
        "--output", default="prototype.html",
        help="Fichier de sortie HTML (défaut : prototype.html)"
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"❌ PDF introuvable : {pdf_path}")
        return

    print(f"\n🚀 Démarrage du pipeline sur : {pdf_path.name}")
    state = run_pipeline(str(pdf_path))

    if state.get("final_result") == "ERROR" or not state.get("html_code"):
        print("\n❌ Le pipeline a échoué. Erreurs :")
        for err in state.get("errors", []):
            print(f"   - {err}")
        return

    output_path = Path(args.output)
    output_path.write_text(state["html_code"], encoding="utf-8")
    print(f"\n✅ Prototype généré → {output_path}")
    print(f"   Hauteur estimée  : {state.get('render_height', '?')}px")


if __name__ == "__main__":
    main()
