# A - Využití strojového učení pro zpracování InSAR dat

## Seznam členů

- Jan Buriánek (HonzaBuri)

## Zadání

Projekt se zaměřuje na využití metod strojového učení při zpracování dat získaných technologií InSAR. Konkrétně se 
soustředí na fázi zpracování označovanou jako phase unwrapping. Tato fáze patří k nejnáročnějším krokům celého InSAR
workflow, protože jejím cílem je správné přiřazení period 2π tak, aby na sebe jednotlivé části fáze plynule navazovaly
a výsledkem byl spojitý fázový obraz.

Vzhledem k omezenému množství kvalitních reálných interferogramů (tj. dat odrážejících skutečnou topografii nebo 
zemského povrchu) je součástí projektu také vývoj generátoru syntetických dat. Tento generátor bude vytvářet dvojici
snímků: syntetickou unwrapped fázi (label) a odpovídající wrapped fázi (input). Cílem je navrhnout takový generátor,
který bude dostatečně robustní a realistický, aby věrně reprezentoval vlastnosti skutečných dat.

Další klíčovou částí projektu je návrh a implementace vlastní konvoluční neuronové sítě určené pro úlohu
phase unwrapping. Součástí práce bude také analýza a porovnání existujících neuronových sítí vyvíjených pro stejný účel
a jejich srovnání s tradičními metodami používanými pro unwrapping.

Cíle projektu:

- Vytvořit generátor syntetických snímků wrapped (resp. unwrapped) fáze.
- Navrhnout a implementovat konvoluční neuronovou síť schopnou provádět phase unwrapping.
- Porovnat navržené řešení s existujícími metodami (ML i konvenčními).