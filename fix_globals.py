import sys
SRC = 'fast_server.py'
with open(SRC, encoding='utf-8') as f:
    code = f.read()
orig = len(code)

# Fix: ajouter "global compliance_criteria" au debut de chaque endpoint
fixes = [
    # compliance_list
    (
        'async def compliance_list(\n    category: str = Query(None, description="Filtrer par categorie"),\n    mandatory: str = Query(None, description="Yes ou No"),\n    limit: int = Query(500, description="Nombre max de criteres")\n):\n    """Liste les criteres de conformite extraits. Filtrage par category et mandatory."""\n    results = compliance_criteria',
        'async def compliance_list(\n    category: str = Query(None, description="Filtrer par categorie"),\n    mandatory: str = Query(None, description="Yes ou No"),\n    limit: int = Query(500, description="Nombre max de criteres")\n):\n    """Liste les criteres de conformite extraits. Filtrage par category et mandatory."""\n    global compliance_criteria\n    results = compliance_criteria'
    ),
    # compliance_stats
    (
        'async def compliance_stats():\n    """Statistiques sur les criteres extraits par categorie."""\n    by_cat = {}',
        'async def compliance_stats():\n    """Statistiques sur les criteres extraits par categorie."""\n    global compliance_criteria\n    by_cat = {}'
    ),
    # compliance_export
    (
        'async def compliance_export(fmt: str = Query("csv", description="Format: csv ou excel")):\n    """\n    if not compliance_criteria:',
        'async def compliance_export(fmt: str = Query("csv", description="Format: csv ou excel")):\n    global compliance_criteria\n    """\n    if not compliance_criteria:'
    ),
]

for old, new in fixes:
    if old in code:
        code = code.replace(old, new, 1)
        print(f'Fixed: {old[:60].strip()!r}...')
    else:
        # Essai avec version sans docstring intermediaire
        print(f'NOT FOUND: {old[:60].strip()!r}...')

# Verifier syntaxe
import py_compile, os
tmp = SRC + '.chk.py'
with open(tmp, 'w', encoding='utf-8') as f:
    f.write(code)
try:
    py_compile.compile(tmp, doraise=True)
    os.remove(tmp)
    print('SYNTAX OK')
    with open(SRC, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f'DONE: {orig} -> {len(code)} bytes')
except py_compile.PyCompileError as e:
    os.remove(tmp)
    print(f'SYNTAX ERROR: {e}')
    sys.exit(1)
