import os
import pandas as pd
import pyreadstat

# Base directory for raw data: ../datos/raw
base_raw = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datos', 'raw'))
if not os.path.exists(base_raw):
    base_raw = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datos', 'raw'))

print(f"Loading data from: {base_raw}")

# 1. 2003 SAS metadata
_, meta_2003 = pyreadstat.read_sas7bdat(os.path.join(base_raw, 'estudio-nacional-salud-mental-2003-interno', 'adultosfinal.sas7bdat'), metadataonly=True)
df_03 = pd.DataFrame({'var': meta_2003.column_names, 'label': meta_2003.column_labels})
df_03['label_clean'] = df_03['label'].fillna('').astype(str).str.replace('@B', '').str.replace('@/@/', ' - ').str.replace('@U', '').str.replace('@W', '').str.strip()

# 2. 2015 Excel dictionaries
xl_15 = pd.ExcelFile(os.path.join(base_raw, 'base-datos-salud-mental-2015-msps', 'Base de datos y diccionario', 'Diccionario ensm 2015_OK.xlsx'))
dfs_15 = []
for sh in ['Adultos_I', 'Adultos_II', 'Personas', 'Hogares']:
    tmp = pd.read_excel(xl_15, sheet_name=sh)
    text_col = 'Enunciado ' if 'Enunciado ' in tmp.columns else ('Pregunta' if 'Pregunta' in tmp.columns else tmp.columns[1])
    dfs_15.append(pd.DataFrame({
        'tabla': sh,
        'var': tmp['Nombre Campo'],
        'label': tmp[text_col] if text_col in tmp.columns else tmp.iloc[:, 1]
    }))
df_15 = pd.concat(dfs_15, ignore_index=True)
df_15['label_clean'] = df_15['label'].fillna('').astype(str).str.strip()

# 3. 2025 Excel dictionary
df_25 = pd.read_excel(os.path.join(base_raw, 'microdatos-salud-mental-2025deyd', 'Diccionario Bases Etiquetadas y Anonimizadas.xlsx'), sheet_name='Diccionario de datos', header=1)
df_25_clean = pd.DataFrame({
    'tabla': df_25['Id. Tabla'],
    'var': df_25['ID de la variable'],
    'label': df_25['Nombre de la variable o columna']
})
df_25_clean['label_clean'] = df_25_clean['label'].fillna('').astype(str).str.strip()

print(f"Diccionarios cargados: 2003={len(df_03)}, 2015={len(df_15)}, 2025={len(df_25_clean)}")

# Detailed thematic mapping
thematic_areas = [
    {
        'category': '1. TRASTORNO DEPRESIVO MAYOR Y DISTIMIA (CIDI)',
        'items': [
            ('Tristeza o depresión ≥ 2 semanas', 'D24a', 'd24a', 'i2_1_p3_a / i2_1_p3a_a'),
            ('Anhedonia / Pérdida de interés en actividades', 'D25', 'd25', 'i2_1_p3_d / i2_1_p3a_d'),
            ('Sentimiento de inutilidad, culpa o desesperanza', 'D26g', 'd26g', 'i2_1_p3_c / i2_1_p3_e'),
            ('Pérdida de energía / fatiga constante', 'D26b', 'd26b', 'i2_1_p4_a'),
            ('Problemas de sueño (insomnio / hipersomnia)', 'D26h, D26hh', 'd26h, d26hh', 'i2_1_p5_a, i2_1_p5_b'),
            ('Cambios de apetito / peso', 'D26c, D26e', 'd26c, d26e', 'i2_1_p6_a, i2_1_p6_b'),
            ('Dificultad para concentrarse / tomar decisiones', 'D26i', 'd26i', 'i2_1_p4_a'),
            ('Pensamientos recurrentes de muerte', 'D26bb', 'd26bb', 'i2_1_p4_b'),
            ('Edad de inicio del primer episodio depresivo', 'D37', 'd37', 'i2_1_p9'),
            ('Número de episodios depresivos en la vida', 'D38', 'd38', 'i2_1_p10'),
            ('Tratamiento / consulta médica por depresión', 'D41', 'd41', 'i2_1_p17 / i2_13_p238'),
        ]
    },
    {
        'category': '2. CONDUCTA SUICIDA Y AUTOLESIÓN',
        'items': [
            ('Ideación suicida (pensó en suicidarse)', 'D26cc / SC22', 'd26cc / d28a', 'i2_2_p20 (vida) / i2_2_p21 (12m)'),
            ('Plan de suicidio (elaboró un plan)', 'D26dd', 'd26dd', 'i2_2_p22 (vida)'),
            ('Intento de suicidio (intentó suicidarse)', 'D26ee', 'd26ee / intento_suici', 'i2_2_p23 (vida) / i2_2_p24'),
            ('Edad del primer intento de suicidio', 'D29a', 'd29a', 'i2_2_p39'),
            ('Número total de intentos de suicidio en la vida', 'D29', 'd29', 'i2_2_p38'),
            ('Atención médica u hospitalización por intento', 'D29b', 'd29b', 'i2_2_p41'),
            ('Autolesión no suicida (cortarse, quemarse sin intención de morir)', 'IED25 (relacionado)', 'm11_ap (relacionado)', 'i2_2_p45'),
        ]
    },
    {
        'category': '3. TRASTORNO DE ANSIEDAD GENERALIZADA (TAG / GAD)',
        'items': [
            ('Periodo de 6 meses o más preocupado o ansioso', 'SC26b / G1 / G2', 'g1 / g2 / g26', 'i2_4_p76_a / i2_4_p77'),
            ('Dificultad para controlar la preocupación', 'G10a', 'g10a', 'i2_4_p78'),
            ('Inquietud, tensión o dificultad para relajarse', 'G13a', 'g13a', 'i2_4_p80_a'),
            ('Fatiga fácil o cansancio por la preocupación', 'G13b', 'g13b', 'i2_4_p80_b'),
            ('Dificultad para concentrarse / mente en blanco', 'G13c', 'g13c', 'i2_4_p80_c'),
            ('Irritabilidad por la ansiedad', 'G13d', 'g13d', 'i2_4_p80_d'),
            ('Tensión muscular o dolores', 'G13e', 'g13e', 'i2_4_p80_e'),
            ('Problemas de sueño por la ansiedad', 'G13f', 'g13f', 'i2_4_p80_f'),
            ('Edad de inicio de la ansiedad generalizada', 'G26', 'g26', 'i2_4_p84'),
        ]
    },
    {
        'category': '4. ATAQUES DE PÁNICO Y AGORAFOBIA',
        'items': [
            ('Ataque de pánico / crisis repentina de miedo o angustia', 'SC20 / P1', 'pd1 / d_pds12', 'i2_5_p93 (vida) / i2_5_p94 (12m)'),
            ('Palpitaciones / taquicardia durante la crisis', 'P2a', 'pd2a', 'i2_5_p97_a'),
            ('Dificultad para respirar / sensación de ahogo', 'P2b', 'pd2b', 'i2_5_p97_b'),
            ('Miedo a morir o a perder el control / enloquecer', 'P2c, P2d', 'pd2c, pd2d', 'i2_5_p97_c, i2_5_p97_d'),
            ('Agorafobia (miedo a multitudes, transportes, espacios abiertos)', 'A1 / SC20_2', 'd_hyp / m16_', 'i2_6_p111_a, i2_6_p111_b'),
        ]
    },
    {
        'category': '5. FOBIA SOCIAL / ANSIEDAD SOCIAL',
        'items': [
            ('Miedo intenso a hablar en público o actuar frente a otros', 'SC29 / SO1', 'so1 / d_so12', 'i2_7_p125_b'),
            ('Miedo a conocer gente nueva o interactuar socialmente', 'SO2 / SO3', 'so2 / so3', 'i2_7_p125_a'),
            ('Miedo a comer, beber o escribir frente a otras personas', 'SO4 / SO5', 'so4 / so5', 'i2_7_p125_c'),
            ('Evitación activa de situaciones sociales por vergüenza', 'SO9', 'so9', 'i2_7_p127'),
        ]
    },
    {
        'category': '6. TRASTORNO BIPOLAR Y EPISODIO MANÍACO',
        'items': [
            ('Periodo con ánimo anormalmente elevado, eufórico o irritable', 'M1 / M2', 'd_man12 / d_bipolarI12', 'i2_3_p56'),
            ('Disminución de la necesidad de dormir (dormir poco sin cansancio)', 'M3b', 'd_man', 'i2_3_p58_b'),
            ('Hablar mucho más de lo normal / fuga de ideas', 'M3c', 'd_man', 'i2_3_p58_c'),
            ('Aumento de actividades de riesgo / gastos excesivos', 'M3f', 'd_man', 'i2_3_p58_f'),
        ]
    },
    {
        'category': '7. TRAUMA Y ESTRÉS POSTRAUMÁTICO (TEPT / PTSD)',
        'items': [
            ('Exposición a violencia armada, combate, secuestro o amenaza de muerte', 'PTSD1 / DA10', 'm18_p / eventraum', 'i2_8_p140_c, i2_8_p140_d'),
            ('Víctima de asalto físico, agresión grave o atraco a mano armada', 'PTSD2', 'm18_p', 'i2_8_p140_g'),
            ('Víctima de violencia sexual o abuso', 'PTSD3', 'm18_p', 'i2_8_p140_h'),
            ('Recuerdos intrusivos, pesadillas o reviviscencias del trauma', 'PTSD10', 'm18_p', 'i2_8_p151_a, i2_8_p151_b'),
            ('Evitación de pensamientos, lugares o personas asociadas al trauma', 'PTSD11', 'm18_p', 'i2_8_p151_c'),
            ('Hipervigilancia, sobresaltos o tensión constante tras el trauma', 'PTSD12', 'm18_p', 'i2_8_p151_d'),
        ]
    },
    {
        'category': '8. CONSUMO DE ALCOHOL Y PROBLEMAS RELACIONADOS',
        'items': [
            ('Consumo de alcohol alguna vez en la vida', 'SU1', 'bebe', 'j6_p81'),
            ('Frecuencia habitual de consumo de alcohol', 'SU2', 'audit / m13a_p1', 'j6_p82'),
            ('Consumo excesivo ocasional (binge drinking / 5+ tragos)', 'SU3', 'audit_c / m13a_p3', 'j6_p84'),
            ('Incapacidad para parar de beber una vez empezado', 'SU11', 'm13a_p4', 'j6_p87'),
            ('Remordimiento o culpa después de beber', 'SU13', 'm13a_p7', 'j6_p90'),
            ('Lagunas mentales / no recordar lo sucedido por beber', 'SU14', 'm13a_p8', 'j6_p91'),
            ('Herirse o herir a otros como resultado del consumo', 'SU15', 'm13a_p9', 'j6_p92'),
        ]
    },
    {
        'category': '9. CONSUMO DE TABACO Y CIGARRILLO',
        'items': [
            ('Fumador actual de cigarrillo / tabaco', 'SC7 / TB1', 'm11_bp1', 'j6_p45 / j6_p47'),
            ('Edad a la que empezó a fumar diariamente', 'TB2', 'm11_bp2', 'j6_p49'),
            ('Cantidad de cigarrillos consumidos al día', 'TB3', 'm11_bp3', 'j6_p51'),
            ('Tiempo transcurrido desde que despierta hasta el primer cigarrillo', 'TB4', 'm11_bp4', 'j6_p53'),
        ]
    },
    {
        'category': '10. CONSUMO DE SUSTANCIAS ILÍCITAS',
        'items': [
            ('Consumo de Marihuana / Cannabis en la vida / 12 meses', 'SU40a', 'm11_bp10', 'j6_p106 / j6_p107'),
            ('Consumo de Cocaína en la vida / 12 meses', 'SU40b', 'm11_bp11', 'j6_p121 / j6_p122'),
            ('Consumo de Bazuco / Crack en la vida / 12 meses', 'SU40c', 'm11_bp12', 'j6_p129 / j6_p130'),
            ('Uso no médico de tranquilizantes / sedantes', 'SU40d / PH2', 'm11_bp13', 'j6_p148 / j6_p149'),
            ('Edad de primer consumo de sustancias', 'SU41', 'm11_bp10es', 'j6_p109, j6_p124'),
        ]
    },
    {
        'category': '11. UTILIZACIÓN DE SERVICIOS DE SALUD MENTAL Y BARRERAS',
        'items': [
            ('Consulta por salud mental / problemas emocionales (vida / 12m)', 'SR1 / SR2', 'm24_p1', 'k1_p1 / k1_p2'),
            ('Profesional consultado: Psiquiatra', 'SR11a', 'm24_p2a', 'k1_p4_a'),
            ('Profesional consultado: Psicólogo', 'SR11b', 'm24_p2b', 'k1_p4_b'),
            ('Profesional consultado: Médico general', 'SR11c', 'm24_p2c', 'k1_p4_c'),
            ('Profesional consultado: Trabajador social / Terapeuta', 'SR11d', 'm24_p2d', 'k1_p4_d'),
            ('Profesional consultado: Consejero espiritual / Religioso', 'SR11e', 'm24_p2e', 'k1_p4_e'),
            ('Hospitalización por problemas de salud mental', 'SR20', 'm24_p10', 'k1_p14'),
            ('Barrera de acceso: Costo económico / falta de dinero', 'SR30a', 'm24_p15a', 'k1_p18_a'),
            ('Barrera de acceso: Falta de aseguramiento / problema EPS', 'SR30b', 'm24_p15b', 'k1_p18_b'),
            ('Barrera de acceso: Pensó que se resolvería solo', 'SR30c', 'm24_p15c', 'k1_p18_c'),
            ('Barrera de acceso: Miedo al estigma / qué dirán los demás', 'SR30d', 'm24_p15d', 'k1_p18_d'),
            ('Barrera de acceso: Demora en citas / citas muy lejanas', 'SR30e', 'm24_p15e', 'k1_p18_e'),
        ]
    },
    {
        'category': '12. MEDICAMENTOS PSICOTRÓPICOS FORMULADOS',
        'items': [
            ('Toma o tomó medicamentos formulados para los nervios/ánimo', 'PH1', 'Medicamentos_adultos', 'k2_p22'),
            ('Uso de antidepresivos', 'PH2_1b', 'Medicamentos_adultos', 'k2_p23_b'),
            ('Uso de tranquilizantes / ansiolíticos', 'PH2_1a', 'Medicamentos_adultos', 'k2_p23_a'),
            ('Uso de medicamentos para dormir (somníferos)', 'PH2_1c', 'Medicamentos_adultos', 'k2_p23_c'),
        ]
    },
    {
        'category': '13. ESTADO DE SALUD PERCIBIDA Y ENFERMEDADES CRÓNICAS',
        'items': [
            ('Autopercepción del estado de salud general (Excelente a Mala)', 'SC9a / SC8_1', 'Estados_de_Salud / m5_p1', 'c1_p15 / c1_p16'),
            ('Diagnóstico de Hipertensión arterial', 'CC1', 'm7_p1', 'c3_p33_c'),
            ('Diagnóstico de Diabetes', 'CC2', 'm7_p2', 'c3_p33_d'),
            ('Diagnóstico de Enfermedad respiratoria crónica / Asma', 'CC3', 'm7_p3', 'c3_p33_a'),
        ]
    },
    {
        'category': '14. DISCAPACIDAD Y DÍAS FUERA DE ROL (WHODAS / SHEEHAN)',
        'items': [
            ('Días en los últimos 30 días totalmente incapaz de trabajar/estudiar', 'FD1', 'm20_p1', 't0_p30 / t0_p31'),
            ('Días con rendimiento reducido a la mitad o menos', 'FD2', 'm20_p2', 't0_p32'),
            ('Dificultad para concentrarse / recordar información', 'FD10', 'm20_p10', 't0_c52 / Washington Group'),
            ('Dificultad para caminar o subir escaleras', 'FD11', 'm20_p11', 't0_c51'),
            ('Dificultad en el autocuidado (bañarse / vestirse)', 'FD12', 'm20_p12', 't0_c53'),
            ('Dificultad para relacionarse con personas y participar en comunidad', 'FD18', 'm20_p18', 't0_c55'),
        ]
    },
    {
        'category': '15. CARACTERÍSTICAS SOCIODEMOGRÁFICAS Y ECONÓMICAS ADICIONALES',
        'items': [
            ('Estado conyugal / Estado civil (casado, unión libre, soltero, viudo, separado)', 'SC3 / SC3a', 'EstadoCivil', 'c1_p6 / c1_p7'),
            ('Condición de actividad laboral (ocupado, desocupado, inactivo)', 'EM1 / EM2', 'hr_trab / m2_ocupacion', 'e1_p1 / e1_p2'),
            ('Afiliación al Sistema de Seguridad Social en Salud (Régimen Contributivo / Subsidiado)', 'DA1_9', 'Personas / Hogares', 'c2_p24 / c2_p25'),
            ('Jefatura del hogar (jefe/a de hogar)', 'INT_PER / HU10', 'Hogares (parentesco)', 'c1_p2 (parentesco con jefe)'),
            ('Tamaño del hogar (número de personas en el hogar)', 'INT_MEN1-6 / HU9', 'Hogares (num_personas)', 'Hogares (conteo personas)'),
        ]
    }
]

# Print markdown summary
for cat in thematic_areas:
    print(f"\n### {cat['category']}")
    print("| Concepto / Pregunta Armonizable | ENSM 2003 (Blaise/SAS) | ENSM 2015 (CAPI/CSV) | ENSM 2025 (SPSS .sav) |")
    print("| :--- | :--- | :--- | :--- |")
    for item in cat['items']:
        print(f"| **{item[0]}** | `{item[1]}` | `{item[2]}` | `{item[3]}` |")
