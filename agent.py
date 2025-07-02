import streamlit as st
import re
import os
import openai
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
import requests
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
import xml.etree.ElementTree as ET
import urllib.parse

st.markdown("""
<style>
/* Verhindert, dass Buttons umbrechen */
.stButton>button {
  white-space: nowrap;
}
/* Optional: etwas Abstand und fixierte Position */
.back-btn-container {
  position: absolute;
  top: 1rem;
  left: 1rem;
}
</style>
""", unsafe_allow_html=True)


I18N = {
    "de": {
        "welcome_title": "Hi, ich bin Savie",
        "welcome_body": """### Hi, ich bin **Savie**,  
**dein Begleiter bei Saventic Care**! 🌟  

Ich helfe dir gerne bei <strong>allgemeinen Fragen</strong> zu unserem Service (kein Diagnose-Service).  
Außerdem gebe ich <strong>allgemeine Informationen</strong> zu seltenen Erkrankungen (keine persönliche Diagnose).  

<strong>Wichtig:</strong>  
<em>Bei medizinischen Notfällen oder akuten Beschwerden wende dich bitte direkt an einen Arzt oder den Notruf.</em> 😊
<strong>Hinweis zur Datenverarbeitung:</strong>  
Diese Anwendung wird unterstützt durch <strong>OpenAI</strong>. Deine Eingaben werden zur Generierung der Antworten verarbeitet.  
Informationen zu seltenen Erkrankungen stammen aus <strong>Orphadata</strong>, einer öffentlichen Datenbank.  

<strong>Haftungsausschluss:</strong>  
Trotz sorgfältiger Recherche können die bereitgestellten Informationen ungenau, unvollständig oder veraltet sein. Bitte prüfe alle Antworten eigenständig und konsultiere bei Unsicherheiten Fachpersonen.  

Bitte wähle unten aus: Frage **zum Service** oder Frage **zu seltener Erkrankung**.""",
        "help_button": "",
        "chat_input_placeholder": "Stelle eine Frage…",
        "typing": "Savie tippt…",
        "med_response": "Ich kann dir bei medizinischen Einschätzungen leider nicht weiterhelfen. Du kannst unser medizinisches Formular ausfüllen.",
        "form_btn": "Formular ausfüllen",
        "support_btn": "Patientensupport verständigen",
        "faq_prompt": """Du bist **Savie**, ein freundlicher Begleiter bei Saventic Care.  
Nutze **nur** die folgenden Auszüge aus unseren Service-Dokumenten (FAQ).
Antworte **nur** mit dem reinen Antworttext und entferne das Präfix „A:“.
Wenn keine Antwort gefunden wird, antworte: „Dazu habe ich leider keine Informationen.“  

**Dokumente:**  
{context}  

**Frage:**  
{question}  

**Antwort (in Du-Form):**""",
        "no_info_response": "Dazu habe ich leider keine Informationen.",
        "cta_text": "Möchtest du noch mehr über diese Erkrankung erfahren? Klicke auf den Button, um weitere Infos zu den Symptomen zu bekommen.",
        "button_text": "Mehr erfahren 😺",
        "more_info_btn": "Symptome anzeigen 😺",
        "select_subtype_prompt": "Bitte wähle einen Subtyp aus:",
        "subtype_submit_btn": "Anzeigen",
        "service_button": "Frage zum Service",
        "rare_button": "Frage zu seltener Erkrankung",
        "select_mode_prompt": "Wähle bitte zuerst einen Modus aus.",
        "switch_mode_btn": "← Anderen Modus wählen",
        "lang_mismatch_msg": "Ich habe erkannt, dass deine Frage auf {detected} gestellt wurde. Möchtest du zur {detected}-Oberfläche wechseln oder auf {current} weitermachen?",
        "mismatch_continue_btn": "Auf {current} weitermachen",
        "mismatch_switch_btn": "Wechsel zu {detected}",
        # Häufigkeits-Texte
        "freq_very_frequent": "Sehr häufig",
        "freq_frequent": "Häufig",
        "freq_occasional": "Gelegentlich",
        "summary_symptoms": "Zu dieser Erkrankung sind {vf} sehr häufige, {f} häufige und {o} gelegentliche Symptome beschrieben.",
    },
    "en": {
        "welcome_title": "Hi, I’m Savie",
        "welcome_body": """### Hi, I’m **Savie**,  
**your companion at Saventic Care**! 🌟  

I can help you with <strong>general questions</strong> about our service (no diagnoses).  
I also provide <strong>general information</strong> on rare diseases (no personal diagnosis). 

<strong>Important:</strong>  
<em>For medical emergencies or acute issues, please contact a doctor or emergency services immediately.</em> 😊
<strong>Data Processing Notice:</strong>  
This application is powered by <strong>OpenAI</strong>. Your inputs are processed to generate responses.  
Information about rare diseases is sourced from <strong>Orphadata</strong>, a public database.  

<strong>Disclaimer:</strong>  
Despite careful research, the information provided may be inaccurate, incomplete, or outdated. Please verify all answers yourself and consult professionals if in doubt.


Please choose below: Service question or Rare disease question.""",
        "help_button": "",
        "chat_input_placeholder": "Ask a question…",
        "typing": "Savie is typing…",
        "med_response": "I’m sorry, but I can’t provide medical advice. You can fill out our medical information form.",
        "form_btn": "Fill out form",
        "support_btn": "Contact patient support",
        "faq_prompt": """You are **Savie**, your friendly companion at Saventic Care.  
Use **only** the following excerpts from our service FAQ documents.
Respond **only** with the answer text and remove the 'A:' prefix.  
If you find no answer, reply: “I’m sorry, I have no information on that.”  

**Documents:**  
{context}  

**Question:**  
{question}  

**Answer (in friendly tone):**""",
        "no_info_response": "I’m sorry, I have no information on that.",
        "cta_text": "Would you like to learn more about the symptoms of this disease? Click the button to get additional info.",
        "button_text": "Learn more 😺",
        "more_info_btn": "Show me the symptoms 😺",
        "select_subtype_prompt": "Please select a subtype:",
        "subtype_submit_btn": "Show",
        "service_button": "Service question",
        "rare_button": "Rare disease question",
        "select_mode_prompt": "Please select a mode first.",
        "switch_mode_btn": "← Switch mode",
        "lang_mismatch_msg": "I detected your question was in {detected}. Switch to {detected} or continue in {current}?",
        "mismatch_continue_btn": "Continue in {current}",
        "mismatch_switch_btn": "Switch to {detected}",
        # Häufigkeits-Texte
        "freq_very_frequent": "Very frequent",
        "freq_frequent": "Frequent",
        "freq_occasional": "Occasional",
        "summary_symptoms": "For this disease, there are {vf} very frequent, {f} frequent, and {o} occasional symptoms described.",
    },
    "pl": {
        "welcome_title": "Cześć, jestem Savie",
        "welcome_body": """### Cześć, jestem **Savie**,  
**Twoim przewodnikiem w Saventic Care**! 🌟  

Mogę pomóc w <strong>ogólnych pytaniach</strong> dotyczących naszego serwisu (bez diagnoz).  
Również udzielam <strong>ogólnych informacji</strong> o rzadkich chorobach (bez osobistej diagnozy).  

<strong>Ważne:</strong>  
<em>W razie nagłych wypadków medycznych lub ostrych dolegliwości skontaktuj się od razu z lekarzem lub pogotowiem.</em> 😊
<strong>Informacja o przetwarzaniu danych:</strong>  
Aplikacja jest wspierana przez <strong>OpenAI</strong>. Twoje dane wejściowe są przetwarzane w celu generowania odpowiedzi.  
Informacje o rzadkich chorobach pochodzą z <strong>Orphadata</strong>, publicznej bazy danych.  

<strong>Zrzeczenie się odpowiedzialności:</strong>  
Pomimo starannego gromadzenia informacji dostarczone dane mogą być nieścisłe, niekompletne lub nieaktualne. Prosimy o samodzielne weryfikowanie odpowiedzi i konsultację z ekspertami w razie wątpliwości.


Proszę wybierz: pytanie **o serwis** lub pytanie **o rzadką chorobę**.""",
        "help_button": "",
        "chat_input_placeholder": "Zadaj pytanie…",
        "typing": "Savie pisze…",
        "med_response": "Przepraszam, ale nie udzielam porad medycznych. Możesz wypełnić nasz formularz medyczny.",
        "form_btn": "Wypełnij formularz",
        "support_btn": "Skontaktuj się z pomocą pacjenta",
        "faq_prompt": """Jesteś **Savie**, przyjaznym przewodnikiem Saventic Care.  
Użyj **wyłącznie** poniższych fragmentów naszych dokumentów FAQ.
Odpowiadaj **tylko** czystym tekstem odpowiedzi i usuń prefiks „A:“. 
Jeśli nie znajdziesz odpowiedzi, odpowiedz: „Niestety nie mam na ten temat informacji.”  

**Dokumenty:**  
{context}  

**Pytanie:**  
{question}  

**Odpowiedź (w formie nieformalnej):**""",
        "no_info_response": "Niestety nie mam na ten temat informacji.",
        "cta_text": "Chcesz dowiedzieć się jeszcze więcej o tej chorobie? Kliknij poniżej, aby zobaczyć dodatkowe informacje o symptomach.",
        "button_text": "Więcej informacji 😺",
        "more_info_btn": "Pokaż mi objawy 😺",
        "select_subtype_prompt": "Wybierz podtyp:",
        "subtype_submit_btn": "Pokaż",
        "service_button": "Pytanie o serwis",
        "rare_button": "Pytanie o rzadką chorobę",
        "select_mode_prompt": "Wybierz najpierw tryb.",
        "switch_mode_btn": "← Wybierz inny tryb",
        "lang_mismatch_msg": "Wykryłem, że Twoje pytanie zostało zadane w języku {detected}. Czy chcesz przełączyć interfejs na {detected}, czy kontynuować w {current}?",
        "mismatch_continue_btn": "Kontynuuj w {current}",
        "mismatch_switch_btn": "Przełącz na {detected}",
        # Häufigkeits-Texte
        "freq_very_frequent": "Bardzo częste",
        "freq_frequent": "Częste",
        "freq_occasional": "Okazjonalne",
        "summary_symptoms": "Dla tej choroby opisano {vf} bardzo częstych, {f} częstych i {o} okazjonalnych objawów.",
    },
    "es": {
        "welcome_title": "Hola, soy Savie",
        "welcome_body": """### Hola, soy **Savie**,  
**tu acompañante en Saventic Care**! 🌟  

Puedo ayudarte con <strong>preguntas generales</strong> sobre nuestro servicio (sin diagnósticos).  
También ofrezco <strong>información general</strong> sobre enfermedades raras (sin diagnóstico personal).   


<strong>Importante:</strong>  
<em>Para emergencias médicas o problemas agudos, contacta inmediatamente a un médico o al servicio de emergencias.</em> 😊  
<strong>Aviso sobre el procesamiento de datos:</strong>  
Esta aplicación está respaldada por <strong>OpenAI</strong>. Tus entradas se procesan para generar respuestas.  
La información sobre enfermedades raras proviene de <strong>Orphadata</strong>, una base de datos pública.  

<strong>Aviso legal:</strong>  
A pesar de una investigación cuidadosa, la información proporcionada puede ser inexacta, incompleta o estar desactualizada. Por favor, verifica todas las respuestas por ti mismo y consulta a profesionales si tienes dudas.


Por favor elige: pregunta **sobre el servicio** o pregunta **sobre enfermedad rara**.""",
        "help_button": "",
        "chat_input_placeholder": "Haz una pregunta…",
        "typing": "Savie está escribiendo…",
        "med_response": "Lo siento, pero no puedo dar consejos médicos. Puedes llenar nuestro formulario médico.",
        "form_btn": "Rellenar formulario",
        "support_btn": "Contactar soporte al paciente",
        "faq_prompt": """Eres **Savie**, tu compañero amigable en Saventic Care.  
Usa **solo** los siguientes extractos de nuestros documentos de FAQ.
Responde **solo** con el texto de la respuesta y elimina el prefijo “A:”.  
Si no encuentras respuesta, responde: “Lo siento, no tengo información al respecto.”  

**Documentos:**  
{context}  

**Pregunta:**  
{question}  

**Respuesta (tono amigable):**""",
        "no_info_response": "Lo siento, no tengo información al respecto.",
        "cta_text": "¿Quieres saber más sobre los síntomas de esta enfermedad? Haz clic en el botón para obtener información adicional.",
        "button_text": "Saber más 😺",
        "more_info_btn": "Muéstrame los síntomas 😺",
        "select_subtype_prompt": "Selecciona un subtipo:",
        "subtype_submit_btn": "Mostrar",
        "service_button": "Pregunta sobre el servicio",
        "rare_button": "Pregunta sobre enfermedad rara",
        "select_mode_prompt": "Por favor, selecciona primero un modo.",
        "switch_mode_btn": "← Cambiar modo",
        "lang_mismatch_msg": "He detectado que tu pregunta está en {detected}. ¿Cambiar a {detected} o continuar en {current}?",
        "mismatch_continue_btn": "Continuar en {current}",
        "mismatch_switch_btn": "Cambiar a {detected}",
        "lang_mismatch_msg": "Detectei que sua pergunta foi feita em {detected}. Deseja mudar para {detected} ou continuar em {current}?",
        "mismatch_continue_btn": "Continuar em {current}",
        "mismatch_switch_btn": "Mudar para {detected}",
        # Häufigkeits-Texte
        "freq_very_frequent": "Muy frecuentes",
        "freq_frequent": "Frecuentes",
        "freq_occasional": "Ocasionales",
        "summary_symptoms": "Para esta enfermedad se describen {vf} síntomas muy frecuentes, {f} frecuentes y {o} ocasionales.",
    },
    "pt": {
        "welcome_title": "Oi, eu sou a Savie",
        "welcome_body": """### Oi, eu sou a **Savie**,  
**seu guia na Saventic Care**! 🌟  

Posso ajudar com <strong>perguntas gerais</strong> sobre nosso serviço (sem diagnósticos).  
Também forneço <strong>informações gerais</strong> sobre doenças raras (sem diagnóstico pessoal).   


<strong>Importante:</strong>  
<em>Em caso de emergências médicas ou problemas agudos, contate imediatamente um médico ou serviços de emergência.</em> 😊
<strong>Aviso sobre processamento de dados:</strong>  
Este aplicativo é suportado pela <strong>OpenAI</strong>. Suas entradas são processadas para gerar respostas.  
As informações sobre doenças raras são provenientes do <strong>Orphadata</strong>, um banco de dados público.  

<strong>Isenção de responsabilidade:</strong>  
Apesar de pesquisas rigorosas, as informações fornecidas podem ser imprecisas, incompletas ou desatualizadas. Verifique todas as respostas por conta própria e consulte profissionais em caso de dúvida.


Por favor, escolha: pergunta **sobre o serviço** ou pergunta **sobre doença rara**.""",
        "help_button": "",
        "chat_input_placeholder": "Faça uma pergunta…",
        "typing": "Savie está digitando…",
        "med_response": "Desculpe, mas não forneço conselhos médicos. Você pode preencher nosso formulário médico.",
        "form_btn": "Preencher formulário",
        "support_btn": "Contatar suporte ao paciente",
        "faq_prompt": """Você é **Savie**, seu guia amigável na Saventic Care.  
Use **apenas** os seguintes trechos de nossos documentos de FAQ.  
Responda **apenas** com o texto da resposta e remova o prefixo “A:”.
Se não encontrar resposta, responda: “Desculpe, não tenho informações sobre isso.”  

**Documentos:**  
{context}  

**Pergunta:**  
{question}  

**Resposta (tom amigável):**""",
        "no_info_response": "Desculpe, não tenho informações sobre isso.",
        "cta_text": "Quer saber mais sobre os sintomas desta doença? Clique no botão para obter informações adicionais.",
        "button_text": "Saiba mais 😺",
        "more_info_btn": "Mostre-me os sintomas 😺",
        "select_subtype_prompt": "Selecione um subtipo:",
        "subtype_submit_btn": "Exibir",
        "service_button": "Pergunta sobre o serviço",
        "rare_button": "Pergunta sobre doença rara",
        "select_mode_prompt": "Por favor, selecione primeiro um modo.",
        "switch_mode_btn": "← Mudar modo",
        # Häufigkeits-Texte
        "freq_very_frequent": "Muito frequentes",
        "freq_frequent": "Frequentes",
        "freq_occasional": "Ocasionalmente",
        "summary_symptoms": "Para esta doença, são descritos {vf} sintomas muito frequentes, {f} frequentes e {o} ocasionais.",
    },
}

from faq import FaqTool
from Orphadata_tool import RareDiseaseTool
from orphadata_phenotype_tool import OrphadataPhenotypeTool

import html

# -------------------------------------------------------------------------------
# 1) Patterns für persönliche Diagnose-Anfragen (erste Person, verfeinert)
# -------------------------------------------------------------------------------
PERSONAL_MEDICAL_PATTERNS = {
    "de": re.compile(
        r"\b("
        r"(?:ich habe|habe ich)\b(?: (?:husten|fieber|kopfschmerzen?|bauchschmerzen?|"
        r"rückenschmerzen?|schwindlig|übel|erkrankung|krankheit|schmerzen|symptom|symptome)\b(?: .*)?)"
        r"|bin ich krank\b(?: .*)?"
        r"|mir ist\b(?: .*)?"
        r"|mir tut der? (?:kopf|bauch|rücken|brust) weh\b(?: .*)?"
        r"|ich fühle mich\b(?: .*)?"
        r"|brauche (?:medikament|tablette)\b(?: .*)?"
        r"|was soll ich tun (?:wenn|bei)\b(?: .*)?"
        r")\b",
        re.IGNORECASE
    ),
    "en": re.compile(
        r"\b("
        r"(?:i have|have i)\b(?: (?:cough|fever|headache|stomach ache|"
        r"back pain|nausea|dizzy|illness|disease|symptom|symptoms)\b(?: .*)?)"
        r"|am i sick\b(?: .*)?"
        r"|i feel\b(?: .*)?"
        r"|my (?:head|stomach|back|chest) hurts\b(?: .*)?"
        r"|do i need medicine\b(?: .*)?"
        r"|which medicine should i take\b(?: .*)?"
        r"|what should i do if (?:i have|i feel)\b(?: .*)?"
        r")\b",
        re.IGNORECASE
    ),
    "pl": re.compile(
        r"\b("
        r"(?:mam|czy mam)\b(?: (?:kaszel|gor[ąa]czka|b[óo]l głowy|b[óo]le brzucha|"
        r"b[óo]l plec[óo]w|zawroty głowy|nudno[śs]ci|choroba|objaw|objawy)\b(?: .*)?)"
        r"|boli mnie\b(?: .*)?"
        r"|co mam zrobić jeśli\b(?: .*)?"
        r")\b",
        re.IGNORECASE
    ),
    "es": re.compile(
        r"\b("
        r"(?:tengo|¿tengo)\b(?: (?:tos|fiebre|dolor de cabeza|"
        r"dolor de estómago|dolor de espalda|náuseas|mareado|enfermedad|síntoma|síntomas)\b(?: .*)?)"
        r"|me duele\b(?: .*)?"
        r"|¿qué debo hacer si\b(?: .*)?"
        r")\b",
        re.IGNORECASE
    ),
    "pt": re.compile(
        r"\b("
        r"(?:tenho)\b(?: (?:tosse|febre|dor de cabeça|"
        r"dor de estômago|dor nas costas|náusea|tontura|doença|sintoma|sintomas)\b(?: .*)?)"
        r"|me dói\b(?: .*)?"
        r"|o que devo fazer se\b(?: .*)?"
        r")\b",
        re.IGNORECASE
    ),
}

# -------------------------------------------------------------------------------
# 2) Exception-Patterns: „Ich habe … ausgefüllt …“ o. Ä.
# -------------------------------------------------------------------------------
EXCEPTION_PATTERNS = {
    "de": re.compile(r"\bich habe\b.*\b ausgefüllt\b", re.IGNORECASE),
    "en": re.compile(r"\bi have\b.*\b(filled out|completed)\b", re.IGNORECASE),
    "pl": None,
    "es": None,
    "pt": None,
}

FORM_URL = "https://tst.saventiccare.de/patientendaten-formular/"
SUPPORT_MAIL = "info@saventiccare.de"


# --- State-Schema für den Workflow-Graphen ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ─── Callback für FAQ-Buttons ──────────────────────────────────────────
def _handle_faq(q: str):
    # 1) als User‐Nachricht speichern
    st.session_state.messages.append({"role": "user", "content": q})
    # 2) FAQ abfragen
    result = st.session_state.faq_tool.run({
        "query": q,
        "language": st.session_state.lang,
        "category": None,
        "k": 2
    })
    # 3) Antwort speichern
    answer = result["answer"].strip()
    st.session_state.messages.append({"role": "assistant", "content": answer})
    # 4) neue Suggestions setzen
    st.session_state.suggestions = result.get("suggestions", [])




# Extrahieren
def extract_disease_term(query: str, lang: str) -> str:
    """
    Finds and returns the first disease name in `query`, based on language-specific regexes.
    If nothing matches, returns the original query.
    """
    patterns = {
        "de": r"(?i:\b[A-ZÄÖÜ][\wäöüß-]+(?:-(?:Krankheit|Erkrankung)|\s+(?:Krankheit|Erkrankung|Angioödem))\b)",
        "en": r"(?i:\b[A-Z][\w-]+(?:\s+Disease)\b)|(?i:\b[A-Z][\w-]*angioedema\b)",
        "pl": r"(?i:\b[A-ZŁŻŹÓĄĘĆŃ][\wąćęłńóśźż-]+(?:\s+Choroba)\b)",
        "es": r"(?i:\b[A-ZÁÉÍÓÚÜÑ][\wáéíóúñü-]+(?:\s+Enfermedad)\b)",
        "pt": r"(?i:\b[A-ZÁÉÍÓÚÃÕÇ][\wáéíóúãõç-]+(?:\s+Doença)\b)",
    }
    pat = patterns.get(lang, patterns["en"])
    m = re.search(pat, query)
    return m.group(0) if m else query


# --- OpenAI 1.x-kompatible Übersetzungsfunktion ---
def translate_with_openai(text: str, target_lang: str) -> str:
    name_map = {
        "DE": "Deutsch",
        "PL": "Polnisch",
        "ES": "Spanisch",
        "PT": "Portugiesisch",
        "EN": "Englisch"
    }
    language_name = name_map.get(target_lang, "Englisch")
    prompt = f"Übersetze den folgenden Text ins {language_name}:\n\n{text}"
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI-Übersetzungsfehler:", e)
        return text


def translate_term(term: str, target_lang: str) -> str:
    """
    Übersetze nur das einzelne Wort oder den kurzen Ausdruck,
    ohne zusätzliche Erklärungen.
    """
    language_map = {
        "DE": "Deutsch",
        "EN": "Englisch",
        "ES": "Spanisch",
        "PL": "Polnisch",
        "PT": "Portugiesisch"
    }
    lang_name = language_map.get(target_lang, "Deutsch")
    prompt = (
        f"Übersetze nur das folgende einzelne Wort oder den kurzen Ausdruck ins {lang_name}, "
        f"ohne sonstige Erklärungen oder Beispiele:\n\n"
        f"{term}"
    )
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        # Gib nur die rohe Antwort zurück
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Übersetzungsfehler:", e)
        return term


# --- Chatbot-Node ---
def chatbot(state: State):
    last = state["messages"][-1]
    query = last.content.strip() if hasattr(last, "content") else str(last).strip()
    lang = st.session_state.lang
    texts = I18N[lang]

    # Ausnahme-Pattern für ausgefülltes Formular
    exc_pattern = EXCEPTION_PATTERNS.get(lang)
    if exc_pattern and exc_pattern.search(query):
        result = st.session_state.faq_tool.run(query)
        return {"messages": [AIMessage(content=result["answer"].strip())]}

    # Persönliche Diagnose-Anfrage
    personal_pattern = PERSONAL_MEDICAL_PATTERNS.get(lang, PERSONAL_MEDICAL_PATTERNS["de"])
    if personal_pattern.search(query):
        btn_form = (
            f'<div style="margin-bottom:8px;">'
            f'  <a href="{FORM_URL}" target="_blank">'
            f'    <button style="padding:8px 12px;border:none;border-radius:4px;'
            f'background:#175065;color:white;">{texts["form_btn"]}</button>'
            f'  </a>'
            f'</div>'
        )
        btn_mail = (
            f'<div>'
            f'  <a href="mailto:{SUPPORT_MAIL}">'
            f'    <button style="padding:8px 12px;border:none;border-radius:4px;'
            f'background:#175065;color:white;">{texts["support_btn"]}</button>'
            f'  </a>'
            f'</div>'
        )
        return {
            "messages": [
                AIMessage(content=texts["med_response"]),
                AIMessage(content=btn_form),
                AIMessage(content=btn_mail),
            ]
        }

    # Rare Disease vs. Service Mode in chatbot()
    # 1) Rare Disease Mode: **immer** Orphadata, nie FAQ
    if st.session_state.mode == "rare":
        # --- 1a) Extrahiere den Kerndisease-Term aus der Nutzereingabe ---
        def extract_disease_term(query: str, lang_code: str) -> str:
            patterns = {
                "de": re.compile(
                    r"(?i:\b[A-ZÄÖÜ][\wäöüß-]+"
                    r"(?:(?:-(?:Krankheit|Erkrankung))|\s+(?:Krankheit|Erkrankung|Angioödem))\b)"
                ),
                "en": re.compile(r"(?i:\b[A-Z][\w-]+(?:\s+Disease)\b)|(?i:\b[A-Z][\w-]*angioedema\b)"),
                "pl": re.compile(r"(?i:\b[A-ZŁŻŹÓĄĘĆŃ][\wąćęłńóśźż-]+(?:\s+Choroba)\b)"),
                "es": re.compile(r"(?i:\b[A-ZÁÉÍÓÚÜÑ][\wáéíóúñü-]+(?:\s+Enfermedad)\b)"),
                "pt": re.compile(r"(?i:\b[A-ZÁÉÍÓÚÃÕÇ][\wáéíóúãõç-]+(?:\s+Doença)\b)"),
            }
            pat = patterns.get(lang_code, patterns["en"])
            m = pat.search(query)
            return m.group(0) if m else query  # Fallback: ganzer Query

        # Roh-Query
        query = st.session_state.messages[-1]["content"] if st.session_state.messages else ""

        # Extrahiere Term
        disease_term = extract_disease_term(query, st.session_state.lang)
        st.session_state._last_disease_name = disease_term

        # --- 1b) Übersetze Term ins Englische ---
        eng_term = translate_with_openai(disease_term, "EN")

        # --- 1c) Orphadata-Abfrage mit englischem Term ---
        api_response_en = st.session_state.orphadata_tool.run(eng_term)

        # --- 1d) ORPHAcode extrahieren ---
        m = re.search(r"ORPHAcode:\s*(\d+)", api_response_en)
        st.session_state._last_orpha_number = m.group(1) if m else None

        # --- 1e) Antwort zurück in die Nutzersprache übersetzen ---
        target_lang = st.session_state.lang.upper()
        if target_lang != "EN":
            api_response = translate_with_openai(api_response_en, target_lang)
        else:
            api_response = api_response_en

        return {"messages": [AIMessage(content=api_response)]}

    # 2) Service Mode (FAQ/RAG)
    elif st.session_state.mode == "service":
        # 1) Tool aufrufen
        tool_input = {
            "query": query,
            "language": st.session_state.lang,
            "category": None,
            "k": 2
        }
        result = st.session_state.faq_tool.run(tool_input)

        # 2) Antwort‐Bubble
        msgs = [AIMessage(content=result["answer"].strip())]

        # 3) Buttons für Suggestions als eigene Messages
        if result.get("suggestions"):
            for sug in result["suggestions"]:
                # wir markieren jeden Button-Call als eigenes AIMessage-Objekt
                msgs.append(
                    AIMessage(content=f"__Suggest__: {sug}")
                )
        else:
            msgs.append(AIMessage(content="_Keine weiteren Vorschläge verfügbar._"))

        return {"messages": msgs}

    # 3) Kein Modus gewählt
    else:
        return {"messages": [AIMessage(content=texts["select_mode_prompt"])]}


def _is_no_info(response: str, lang_code: str) -> bool:
    if lang_code == "DE":
        return response.strip() == "Keine Informationen zur angefragten Erkrankung gefunden."
    elif lang_code == "PL":
        return response.strip() == "Brak informacji o tej chorobie."
    elif lang_code == "ES":
        return response.strip() == "No se encontró información sobre esa enfermedad."
    elif lang_code == "PT":
        return response.strip() == "Nenhuma informação encontrada para essa doença."
    else:
        return response.strip() == "No information found for that disease."


# Session-State Defaults (läuft nur einmal)
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.graph_initialized = False
    st.session_state.messages = []
    st.session_state.mode = None
    st.session_state._last_orpha_number = None
    st.session_state._last_disease_name = None
    st.session_state.service_suggestions = []  # ← hier!
    st.session_state.last_suggestion = None


# Diese Defaults immer prüfen
if "show_phenos" not in st.session_state:
    st.session_state.show_phenos = False
if "subtype_map" not in st.session_state:
    st.session_state.subtype_map = None
if "ask_subtype" not in st.session_state:
    st.session_state.ask_subtype = False
#Defaults service mode
if "prompt" not in st.session_state:
    st.session_state.prompt = ""
if "run_faq" not in st.session_state:
    st.session_state.run_faq = False
# ───────── Session‐State Defaults (ganz oben) ─────────
if "service_history" not in st.session_state:
    st.session_state.service_history = []
if "service_suggestions" not in st.session_state:
    st.session_state.service_suggestions = []
if "service_query" not in st.session_state:
    st.session_state.service_query = ""
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []
if "next_query" not in st.session_state:
    st.session_state.next_query = None           # gemerkte Suggestion

config = {"configurable": {"thread_id": "1"}}


def main():
    LANG_MAP = {
        "Deutsch": "de",
        "English": "en",
        "Polski": "pl",
        "Español": "es",
        "Português": "pt",
    }
    if "lang" not in st.session_state:
        choice = st.selectbox(
            "Sprache / Language / Język / Idioma / Língua:",
            list(LANG_MAP.keys())
        )
        if st.button("Continue"):
            st.session_state.lang = LANG_MAP[choice]
            st.session_state.mode = None
            st.session_state.messages = []
            st.session_state._last_orpha_number = None
            st.session_state._last_disease_name = None
            st.rerun()
        return

    lang = st.session_state.lang
    texts = I18N[lang]



    if not st.session_state.graph_initialized:
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        st.session_state.faq_tool = FaqTool(
            persist_directory="./chroma_langchain_db",
            collection_name="example_collection",
            prompt_template=texts["faq_prompt"],
        )

        st.session_state.orphadata_tool = RareDiseaseTool(base_url="https://api.orphadata.com")
        st.session_state.phenotype_tool = OrphadataPhenotypeTool(base_url="https://api.orphadata.com")
        builder = StateGraph(State)
        builder.add_node("chatbot", chatbot)
        builder.add_edge(START, "chatbot")
        builder.add_edge("chatbot", END)
        st.session_state.graph = builder.compile(checkpointer=MemorySaver())
        st.session_state.graph_initialized = True

    # CSS Styling
    st.markdown("""
        <style>
        /* Bestehende Styles */
        [data-testid="stAppViewContainer"] {
          background: linear-gradient(180deg,#D5F5D1 0%,#B3E7F2 100%);
        }
        .message-row { display:flex; align-items:flex-start; margin:10px 0; }
        .user-row { justify-content:flex-end; }
        .assistant-row { justify-content:flex-start; }
        .avatar { width:40px; height:40px; border-radius:50%; margin:0 10px; }
        .message {
          max-width:60%; padding:10px 15px; border-radius:10px;
          box-shadow:0 4px 4px rgba(0,0,0,0.25); animation:slideIn .5s;
        }
        .user-message { background:#175065; color:#fff; }
        .assistant-message { background:#FFFDF3; color:#000; }
        @keyframes slideIn { 0% { transform:translateY(20px); opacity:0; } 100% { transform:translateY(0); opacity:1; } }
         
          /* Dark-Mode-Styles */
        @media (prefers-color-scheme: dark) {
        [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #18454F 0%, #093E32 100%);
          }
        
        /* Chat Layout */
        .message-row     { display: flex; align-items: flex-start; margin: 10px 0; }
        .user-row        { justify-content: flex-end; }
        .assistant-row   { justify-content: flex-start; }
        .avatar          { width: 40px; height: 40px; border-radius: 50%; margin: 0 10px; }
        
      /* Sprechblasen */
      .message {
        max-width: 60%;
        padding: 10px 15px;
        border-radius: 10px;
        box-shadow: 0 4px 4px rgba(0,0,0,0.25);
        animation: slideIn .5s;
      }
      .user-message {
        background: #495054;
        color: #fff;
      }
      .assistant-message {
        background: #23597B;
        color: #fff;
      }
    
      /* Buttons */
      .primary-button, .secondary-button {
        background: #041618;
        color: #fff;
        padding: 0.6em 1.2em;
        border: none;
        border-radius: 8px;
        box-shadow: 0 4px 4px rgba(0,0,0,0.25);
        cursor: pointer;
      }
    
      /* Eingabefeld */
      .message-field {
        background: #495054;
        color: #fff;
        padding: 0.6em 1em;
        border-radius: 8px;
        box-shadow: 0 4px 4px rgba(0,0,0,0.25);
        width: 100%;
      }
    
      /* Tabs-Text auf Weiß setzen */
      [data-testid="stTabs"] [role="tab"],
      [data-testid="stTabs"] [role="tabpanel"] {
        color: #fff !important;
      }
    
      /* Animation */
      @keyframes slideIn {
        0%   { transform: translateY(20px); opacity: 0; }
        100% { transform: translateY(0);    opacity: 1; }
      }

        </style>
    """, unsafe_allow_html=True)


    if st.session_state.mode is None:
        st.markdown(f'''
            <div class="message-row assistant-row">
              <img src="https://raw.githubusercontent.com/hannahleomerx/savie/main/SavieIcon.png" class="avatar">
              <div class="message assistant-message">{texts["welcome_body"]}</div>
            </div>
        ''', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button(texts["service_button"]):
                st.session_state.mode = "service"
                st.session_state.messages = []
                st.session_state._last_orpha_number = None
                st.rerun()
        with col2:
            if st.button(texts["rare_button"]):
                st.session_state.mode = "rare"
                st.session_state.messages = []
                st.session_state._last_orpha_number = None
                st.session_state._last_disease_name = None
                st.rerun()
        st.markdown(
            f"<div style='color:gray; padding:10px;'>{texts['select_mode_prompt']}</div>",
            unsafe_allow_html=True
        )
        return

    st.title(texts["welcome_title"])

    if st.session_state.mode in ("service", "rare"):
        col1, col2 = st.columns([2, 8])
        with col1:
            if st.button(texts["switch_mode_btn"]):
                st.session_state.mode = None
                st.session_state.messages = []
                st.session_state._last_orpha_number = None
                st.session_state._last_disease_name = None
        st.markdown("---")

        # 1) Zuerst alle bisherigen Nachrichten rendern
    for msg in st.session_state.messages:
        content = msg["content"]
        role = msg.get("role", "assistant")
        icon = "USerIcon.png" if role == "user" else "SavieIcon.png"
        disp = html.escape(content) if role == "user" else content
        row, cls = ("user-row", "user-message") if role == "user" else ("assistant-row", "assistant-message")
        st.markdown(f'''
               <div class="message-row {row}">
                 <img src="https://raw.githubusercontent.com/hannahleomerx/savie/main/{icon}" class="avatar">
                 <div class="message {cls}">{disp}</div>
               </div>''', unsafe_allow_html=True)
    # ─────────── Service-Mode ───────────
    if st.session_state.mode == "service":
        # 1) Zeige Chat-History
        for msg in st.session_state.service_history:
            role = msg["role"]
            avatar = "USerIcon.png" if role == "user" else "SavieIcon.png"
            row_cls = "user-row" if role == "user" else "assistant-row"
            msg_cls = "user-message" if role == "user" else "assistant-message"
            content = html.escape(msg["content"]) if role == "user" else msg["content"]
            st.markdown(f'''
              <div class="message-row {row_cls}">
                <img src="https://raw.githubusercontent.com/hannahleomerx/savie/main/{avatar}" class="avatar">
                <div class="message {msg_cls}">{content}</div>
              </div>''', unsafe_allow_html=True)

        # 2) Suggestions-Buttons aus der letzten Antwort
        if st.session_state.suggestions:
            st.markdown("**Weitere Fragen:**")
            for i, sug in enumerate(st.session_state.suggestions):
                if st.button(sug, key=f"svc_sugg_{i}"):
                    st.session_state.next_query = sug
                    st.session_state.suggestions = []
                    st.rerun()

        # 3) Einziger Chat-Input für Service-Mode
        if st.session_state.next_query:
            prompt = st.session_state.next_query
            st.session_state.next_query = None
        else:
            prompt = st.chat_input(texts["chat_input_placeholder"], key="service_input")

        # 4) Sobald prompt da ist: verarbeite es
        if prompt:
            # A) User-Bubble an History hängen
            st.session_state.service_history.append({"role": "user", "content": prompt})
            st.markdown(f'''
              <div class="message-row user-row">
                <img src="https://raw.githubusercontent.com/hannahleomerx/savie/main/USerIcon.png" class="avatar">
                <div class="message user-message">{html.escape(prompt)}</div>
              </div>''', unsafe_allow_html=True)

            # B) Typing-Indikator
            tp = st.empty()
            tp.markdown(f'''
              <div class="message-row assistant-row">
                <img src="https://raw.githubusercontent.com/hannahleomerx/savie/main/SavieIcon.png" class="avatar">
                <div class="message assistant-message">{texts["typing"]}</div>
              </div>''', unsafe_allow_html=True)

            # C) FAQ-Tool aufrufen
            result = st.session_state.faq_tool.run({
                "query": prompt,
                "language": st.session_state.lang,
                "category": None,
                "k": 2
            })
            tp.empty()

            # D) Assistant-Bubble an History hängen
            answer = result["answer"].strip()
            st.session_state.service_history.append({"role": "assistant", "content": answer})
            st.markdown(f'''
              <div class="message-row assistant-row">
                <img src="https://raw.githubusercontent.com/hannahleomerx/savie/main/SavieIcon.png" class="avatar">
                <div class="message assistant-message">{answer}</div>
              </div>''', unsafe_allow_html=True)

            # E) Neue Suggestions merken
            st.session_state.suggestions = result.get("suggestions", [])

            # F) sofort neu rendern, damit alles erscheint
            st.rerun()

        # 5) Danach auf keinen Fall in den Rest des Scripts fallen lassen
        return

    # Klick auf "Mehr Infos" – Haupt-Phänotypen oder Subtypen holen

    if st.session_state.mode == "rare" and st.session_state._last_orpha_number:
        # Öffne links-bündigen Container
        st.markdown('<div class="left-align-container">', unsafe_allow_html=True)

        # 1) Show more symptoms Button
        if st.button(texts["more_info_btn"], key="btn_show_phenos"):
            # 1a) Haupt-Phänotypen abrufen
            main_raw = st.session_state.phenotype_tool.get_phenotypes(
                st.session_state._last_orpha_number,
                lang_code=st.session_state.lang.upper()
            )

            if main_raw:
                # 1b) Wenn Daten vorhanden: direkt rendern
                cats = {"Very frequent": [], "Frequent": [], "Occasional": []}
                for p in main_raw:
                    freq = p.get("HPOFrequency", "").lower()
                    if "very frequent" in freq:
                        cats["Very frequent"].append(p)
                    elif "occasional" in freq:
                        cats["Occasional"].append(p)
                    else:
                        cats["Frequent"].append(p)

                vf, f, o = len(cats["Very frequent"]), len(cats["Frequent"]), len(cats["Occasional"])
                st.markdown(f"**{texts['summary_symptoms'].format(vf=vf, f=f, o=o)}**")
                tabs = st.tabs([
                    f"🟢 {texts['freq_very_frequent']} ({vf})",
                    f"🟡 {texts['freq_frequent']} ({f})",
                    f"🔵 {texts['freq_occasional']} ({o})",
                ])
                for i, key in enumerate(["Very frequent", "Frequent", "Occasional"]):
                    with tabs[i]:
                        items = cats[key]
                        if not items:
                            st.write(texts["no_info_response"])
                        else:
                            lines = []
                            for p in items:
                                name = p["Name"]
                                if st.session_state.lang != "en":
                                    name = translate_term(name, st.session_state.lang.upper())
                                lines.append(f"- {name} ({p['HPOId']})")
                            st.markdown("\n".join(lines))

            else:
                # 1c) Subtypen ermitteln über Cross-Reference
                code = st.session_state._last_orpha_number
                lang = st.session_state.lang.upper()

                # HCH-IDs per JSON abrufen
                resp = requests.get(
                    f"https://api.orphadata.com/rd-classification/orphacodes/{code}/hchids",
                    params={"language": lang},
                    headers={"Accept": "application/json"},
                    timeout=5
                )
                hch_list = resp.json().get("data", {}).get("results", [])

                # Alle 'childs'-Codes sammeln
                child_codes = {c for h in hch_list for c in h.get("childs", [])}

                if not child_codes:
                    st.error("Für diesen Code wurden keine Subtypen gefunden.")
                else:
                    # Alle 'childs'-Codes sammeln
                    child_codes = set()
                    for h in hch_list:
                        child_codes.update(h.get("childs", []))

                    if not child_codes:
                        st.error("Für diesen Code wurden keine Subtypen gefunden.")
                    else:
                        # Per Cross-Reference-Endpoint die Namen holen
                        subtype_map = {}
                        for sub_code in child_codes:
                            # 1) Raw-Response holen
                            cr_resp = requests.get(
                                f"https://api.orphadata.com/rd-cross-referencing/orphacodes/{sub_code}",
                                params={"language": lang},  # das ist EN
                                headers={"Accept": "application/json"},
                                timeout=5
                            )
                            cr = cr_resp.json().get("data", {}).get("results", {})

                            # 2) Englischen Namen extrahieren
                            name_en = (
                                    cr.get("Name")
                                    or cr.get("preferredTerm")
                                    or cr.get("Preferred term")
                                    or str(sub_code)
                            )

                            # 3) Übersetzen in die Nutzersprache
                            target = st.session_state.lang.upper()
                            if target != "EN":
                                name = translate_with_openai(name_en, target)
                            else:
                                name = name_en

                            # 4) Eintrag in Map
                            subtype_map[name] = str(sub_code)

                    # Session-State für Dropdown aktivieren
                    st.session_state.subtype_map = subtype_map
                    st.session_state.ask_subtype = True

        # ───────────────────────────────────────────────────────────────
        # 2) Wenn Subtypen da sind: Dropdown in einem Formular + anzeigen
        # ───────────────────────────────────────────────────────────────
        if st.session_state.get("ask_subtype", False):
            # Bot-Nachricht
            st.markdown(f'''
                <div class="message-row assistant-row">
                    <img src="https://raw.githubusercontent.com/hannahleomerx/savie/main/SavieIcon.png"
                         alt="Savie" class="avatar">
                <div class="message assistant-message">
                    {texts["select_subtype_prompt"]}
                </div>
              </div>
''', unsafe_allow_html=True)

            with st.form("subtype_form"):
                choice = st.selectbox(texts["select_subtype_prompt"].rstrip(":"),
                                      list(st.session_state.subtype_map.keys()))
                submitted = st.form_submit_button(texts["subtype_submit_btn"])

            if submitted:
                sub_code = st.session_state.subtype_map[choice]
                sub_raw = st.session_state.phenotype_tool.get_phenotypes(
                    sub_code,
                    lang_code=st.session_state.lang.upper()
                )

                cats = {"Very frequent": [], "Frequent": [], "Occasional": []}
                for p in sub_raw or []:
                    freq = p.get("HPOFrequency", "").lower()
                    if "very frequent" in freq:
                        cats["Very frequent"].append(p)
                    elif "occasional" in freq:
                        cats["Occasional"].append(p)
                    else:
                        cats["Frequent"].append(p)

                vf, f, o = len(cats["Very frequent"]), len(cats["Frequent"]), len(cats["Occasional"])
                st.markdown(f"**{texts['summary_symptoms'].format(vf=vf, f=f, o=o)}**")
                tabs = st.tabs([
                    f"🟢 {texts['freq_very_frequent']} ({vf})",
                    f"🟡 {texts['freq_frequent']} ({f})",
                    f"🔵 {texts['freq_occasional']} ({o})",
                ])
                for i, key in enumerate(["Very frequent", "Frequent", "Occasional"]):
                    with tabs[i]:
                        items = cats[key]
                        if not items:
                            st.write(texts["no_info_response"])
                        else:
                            lines = []
                            for p in items:
                                name = p["Name"]
                                if st.session_state.lang != "en":
                                    name = translate_term(name, st.session_state.lang.upper())
                                lines.append(f"- {name} ({p['HPOId']})")
                            st.markdown("\n".join(lines))

                # Cleanup
                st.session_state.ask_subtype = False
                st.session_state.subtype_map = None

        # Schließe links-bündigen Container
        st.markdown('</div>', unsafe_allow_html=True)

        # Eingabe und Streaming
    if prompt := st.chat_input(texts["chat_input_placeholder"], key="main_chat_input"):

        #1) Direkt die User-Message im Session-State speichern
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 1a) UND sofort manuell rendern, damit sie direkt sichtbar ist:
        st.markdown(f'''
          <div class="message-row user-row">
            <img src="https://raw.githubusercontent.com/hannahleomerx/savie/main/USerIcon.png" class="avatar">
            <div class="message user-message">{prompt}</div>
          </div>''', unsafe_allow_html=True)


        # ---------------------------------------------------------------

        # 2) Rare-Disease-Modus: Extrahiere ggf. den rohen Begriff
        if st.session_state.mode == "rare":
            st.session_state._last_disease_name = extract_disease_term(
                prompt, st.session_state.lang
            )

        # 3) Zeige „Savie tippt…“
        tp = st.empty()
        tp.markdown(f'''
            <div class="message-row assistant-row">
              <img src="https://raw.githubusercontent.com/hannahleomerx/savie/main/SavieIcon.png" class="avatar">
              <div class="message assistant-message">{texts["typing"]}</div>
            </div>''', unsafe_allow_html=True)

        # 4) Führe deine Chat-Logik aus (streaming)
        user_texts = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
        full = ""
        for update in st.session_state.graph.stream({"messages": user_texts}, config, stream_mode="updates"):
            msgs = update.get("chatbot", {}).get("messages", [])
            for ai in msgs:
                full += ai.content

        # 5) Entferne den „tippt“-Hinweis
        tp.empty()

        # 6) Bot-Antwort speichern
        st.session_state.messages.append({"role": "assistant", "content": full})

        # ORPHAcode nach dem Streaming extrahieren
        #m = re.search(r"ORPHAcode:\s*(\d+)", full)
        #st.session_state._last_orpha_number = m.group(1) if m else None

        # 7) Zum Schluss neu rendern, damit alles zusammen angezeigt wird
        st.rerun()


if __name__ == "__main__":
    main()