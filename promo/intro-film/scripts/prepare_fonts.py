"""Create the static caption font used by libass for burned-in captions.

Usage: python scripts/prepare_fonts.py FONTS_DIR
Instances the OFL DM Sans variable font from site/assets/fonts at weight 500
and optical size 14, then names it "DM Sans" so the ASS style can find it.
"""

import sys
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "site/assets/fonts/dm-sans-latin.woff2"


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    font = instancer.instantiateVariableFont(TTFont(SOURCE), {"wght": 500, "opsz": 14})
    font.flavor = None
    names = font["name"]
    for name_id in (16, 17, 21, 22, 25):
        names.removeNames(nameID=name_id)
    for name_id, value in ((1, "DM Sans"), (2, "Regular"), (4, "DM Sans Medium"), (6, "DMSans-Medium")):
        names.setName(value, name_id, 3, 1, 0x409)
    font.save(out / "DMSans-Medium.ttf")
    (out / "OFL.txt").write_text((REPO / "site/assets/fonts/dmsans-OFL.txt").read_text())
    print(out / "DMSans-Medium.ttf")


if __name__ == "__main__":
    main()
