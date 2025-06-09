import json
import subprocess
import os

def convert_pain_intensity(text):
    """
    Konwertuje tekstowy opis bólu na skalę numeryczną 0-2.
    """
    if not text:
        return 0
        
    # Lista wyrażeń dla każdego poziomu intensywności
    wyrazenia_0 = [
        'nic nie bolało', 'nie czułem bólu', 'nie czułam bólu', 
        'nic mie nie dokuczało', 'czułem się dobrze', 'czułam się dobrze',
        'mam dobre samopoczucie', 'czuje się kwitnąco', 'nic mi dolega',
        'jestem zdrowy', 'jestem zdrowa'
    ]
    
    wyrazenia_1 = [
        'trochę bolało', 'czułem się źle', 'było to nie wygodne',
        'czułam się źle', 'nie było to straszne uczucie',
        'nie było to okropne', 'do wytrzymania'
    ]
    
    wyrazenia_2 = [
        'bardzo bolało', 'to był bardzo silny ból', 'ból był mocny',
        'dojmujący', 'przejmujący', 'paraliżujący', 'promieniujący',
        'ból był okropny'
    ]
    
    text = text.lower()
    
    # Sprawdzanie wyrażeń dla każdego poziomu
    for wyrazenie in wyrazenia_2:
        if wyrazenie.lower() in text:
            return 2
            
    for wyrazenie in wyrazenia_1:
        if wyrazenie.lower() in text:
            return 1
            
    for wyrazenie in wyrazenia_0:
        if wyrazenie.lower() in text:
            return 0
            
    # Domyślnie zwracamy '0' jeśli nie znaleziono żadnego wyrażenia
    return 0

def ask_llama(prompt, model="llama3.2_codahead"):
    """
    Wywołuje model LLaMA (lub inny dostępny w Ollama) za pomocą polecenia CLI: 'ollama run <model>'.
    Zwraca odpowiedź modelu jako string.
    """
    try:
        process = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        response = process.stdout.decode("utf-8")
        return response.strip()
    except subprocess.CalledProcessError as e:
        print(f"Błąd podczas interakcji z Ollama: {e.stderr.decode('utf-8')}")
        return None

def get_answer_for_question(full_text, question):
    """
    Buduje prompt z wczytanym tekstem wywiadu i pytaniem,
    a następnie zwraca odpowiedź z funkcji ask_llama().
    """
    prompt = (
        f"Tekst wywiadu pacjenta:\n"
        f"\"\"\"\n{full_text}\n\"\"\"\n\n"
        f"Na podstawie powyższego wywiadu odpowiedz (po polsku) na pytanie:\n"
        f"\"{question}\"\n\n"
        f"Twoja odpowiedź (zwięźle, w formie tekstu):"
    )
    return ask_llama(prompt)

def eprescriptions(text):
    frazy = [
        "wypisuje receptę",
        "wystawiam receptę",
        "przepisze leki",
        "przepisze pani leki",
        "przepisze panu leki",
        "wypisze leki",
        "wypisze pani leki",
        "wypisze panu leki",
        "recepta",
        "receptę",
        "recepty"
    ]
    for fraza in frazy:
        if fraza.lower() in text.lower():
            return True
    return False

def ereferrals(text):
    frazy = [
        "wypisuje skierowanie",
        "wystawiam skierowanie",
        "przepisze skierowanie",
        "przepisze pani skierowanie",
        "przepisze panu skierowanie",
        "wypisze skierowanie",
        "wypisze pani skierowanie",
        "wypisze panu skierowanie",
        "skierowanie na badania",
        "skierowanie",
        "skierowania"
    ]
    for fraza in frazy:
        if fraza.lower() in text.lower():
            return True
    return False

def ezla(text):
    frazy = [
        "wypisuje zwolnienie",
        "wystawiam zwolnienie",
        "przepisze druk na zwolnienia",
        "przepisze pani zwolnienie",
        "przepisze panu zwolnienie",
        "wypisze zwolnienie",
        "zwolnienie",
        "zwolnienia",
        "l4",
        "l cztery",
        "el cztery"
    ]
    for fraza in frazy:
        if fraza.lower() in text.lower():
            return True
    return False

def create_output_dict(answers, full_text):
    """
    Buduje słownik w oczekiwanej strukturze JSON na podstawie zebranych odpowiedzi.
    Dodano parametr full_text do sprawdzania usług.
    """
    intensity_value = convert_pain_intensity(answers["skala_dolegliwosci"])
    
    # Sprawdzanie występowania usług w tekście
    check_eprescriptions = eprescriptions(full_text)
    check_ereferrals = ereferrals(full_text)
    check_ezla = ezla(full_text)
    
    output_data = {
        "general_ailments": [
            {
                "name": answers["nazwa_objawow"],
                "description": answers["opis_objawow"],
                "intensity": intensity_value,
                "duration": answers["czas_trwania_dolegliwosci"]
            }
        ],
        "drugs": [
            {
                "name": answers["uzywki"],
                "frequency": answers["czestotliwosc_korzystania_z_uzuwek"],
                "allergies": [
                    {
                        "name": answers["nazwa_alergii"],
                        "medications": [
                            {
                                "name": answers["nazywy_pyrzjomowanych_lekow_przeciw_alergii"],
                                "dosage": answers["dawkowanie_przyjmowanych_lekow_na_alerrgie"]
                            }
                        ]
                    }
                ],
                "previous_conditions": [
                    {
                        "condition": answers["choroby_przewlekle"],
                        "medications": [
                            {
                                "name": answers["leki_przyjmowane_na_stale"],
                                "dosage": answers["dawkowanie_przyjmowanych_lekow_na_alerrgie"]
                            }
                        ]
                    }
                ],
                "surgeries_hospitalizations": [
                    {
                        "description": answers["przebyte_zabiegi_i_operacje"],
                        "duration": answers["czas_trwania_leczenia"]
                    }
                ],
                "family_histories": [
                    {
                        "description": answers["choroby_wystepujace_w_rodzinie"]
                    }
                ]
            }
        ],
        "services": {
            "eprescriptions": check_eprescriptions,
            "ereferrals": check_ereferrals,
            "ezla": check_ezla
        }
    }
    return output_data


def inference(text):


    questions = {
        "nazwa_objawow": "Jak się nazywają dolegliwości pacjenta z wywiadu?",
        "opis_objawow": "Jakie były objawy dolegliwości?",
        "skala_dolegliwosci": "Jak dokuczliwe były objawy?",
        "czas_trwania_dolegliwosci": "Jak długo trwają objawy?",
        "choroby_przewlekle": "Czy pacjent ma choroby współistniejące? (czy choruje przewlekle?)",
        "leki_przyjmowane_na_stale": "Czy pacjent przyjmuje jakieś leki na stałe?",
        "nazwa_alergii": "Czy pacjent ma alergie?",
        "nazywy_pyrzjomowanych_lekow_przeciw_alergii": "Jakie leki pacjent przyjmuje na alergię?",
        "dawkowanie_przyjmowanych_lekow_na_alerrgie": "Jakie jest dawkowanie leków na alergię?",
        "uzywki": "Czy pacjent ma jakieś nałogi?",
        "czestotliwosc_korzystania_z_uzuwek": "Jak często pacjent stosuje używki?",
        "choroby_wystepujace_w_rodzinie": "Jakie choroby występowały w przeszłości w rodzinie?",
        "przebyte_zabiegi_i_operacje": "Czy pacjent przechodził jakąś operację lub zabieg w szpitalu lub przychodni?",
        "czas_trwania_leczenia": "Jeżeli pacjent przechodził leczenie w szpitalu, to jak długo ono trwało?"
    }

    answers = {}
    for key, question in questions.items():
        answer = get_answer_for_question(text, question)
        answers[key] = answer

    output_data = create_output_dict(answers, text)

    print(f"Result: {answers}")