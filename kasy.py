'''
Market Piotr i Paweł ma kasy, do któych co X (rozkład) minut przychodzą klienci (jak uwzględnić godziny szczytu)
Jeśli przed klientem do kasy stoi więcej niż 5 klient otrzymuje on 10% zwrot kwoty zakupów. Jeśli klient stoi w kasie
za długo może dojśc do zjawiska zrezygnowania z przyszłych zakupów (jakaś dystrybuanta). Wartość zakupu ma pewien
rozkład, a
w wypadku churnu tracimy klienta na cały horyzont symulacji, gdzie średnia liczba zkaupów w miesiącu i ich wartość
również będzie losowana wg rozkładu

Funkcje:
Symulacja kolejnych przychodzących klientów - do n kas, do których jest wspólna kolejka

Czas obsłługi - skorelowany z wielkością zakupów
Losowanie wartości zakupu
Losowanie odejścia (churn) i parametrów klienta???
Wyznaczanie zwrotu dla klientów, których numer w kolejce jest większy lub równy 6 (w momencie kiedy do niej przychodzi)

 Funkcja odpalająca symulację
'''
import random
import simpy
import scipy.stats
import numpy as np
import pandas as pd

ST_CZAS_OBSLUGI = 10
SZYBKOSC_OBSLUGI = 1 / 20
WARTOSC_ZAKUPOW_SR = 80
WARTOSC_ZAKUPOW_ODCH = 20
INTERWAL_SPOKOJ = 1000
INTERWAL_SZCZYT = 150
SR_LICZBA_KLIENTOW = 100
ODCH_LICZBA_KLIENTOW = 10
HORYZONT = 30  # chyba nie będzie potrzebny
RANDOM_SEED = 42
WSP_ZWROT = 0.10

# parametry powiązane z churn
WSP_UTR_PTN = 0.5  # strata ok. połowy CLV w danym horyzoncie
PRZ_LICZBA_WIZYT_ROK = 36
HORYZONT = 1

UTRACONY_POTENCJAL = WSP_UTR_PTN * PRZ_LICZBA_WIZYT_ROK * HORYZONT * WARTOSC_ZAKUPOW_SR
# parametry rozkładu Weibulla
LAMBDA = 100
K = 3  # wraz ze wzrostem czasu oczekiwania wzrasta p-pstwo rezygnacji klienta z następnych zakupów

klienci_obsluzeni = 0
numer_w_kolejce = 0

liczba_klientow = round(random.normalvariate(SR_LICZBA_KLIENTOW, ODCH_LICZBA_KLIENTOW))

klient_wartosci = pd.DataFrame(
    columns=['id_klienta', 'capacity', 'oczekiwanie', 'wartosc_zakupow', 'zwrot', 'czas_obslugi',
             "utracony_potencjal"], dtype=np.float64)

zrodlo_wartosci = pd.DataFrame(columns=['start_czas', "numer_w_kolejce"], dtype=np.float64)


def zrodlo(env, liczba_klientow, INTERWAL_SPOKOJ, INTERWAL_SZCZYT, kasa, klienci_obsluzeni):
    """Source generates customers randomly"""
    global numer_w_kolejce
    global zrodlo_wartosci
    for i in range(liczba_klientow):
        c = klient('Klient%02d' % i, env, WARTOSC_ZAKUPOW_SR, WARTOSC_ZAKUPOW_ODCH, kasa,
                   numer_w_kolejce, WSP_ZWROT, K, LAMBDA, UTRACONY_POTENCJAL)

        numer_w_kolejce = i - klienci_obsluzeni
        env.process(c)
        if i < 2000 or i > 8000:
            t = random.expovariate(1.0 / INTERWAL_SPOKOJ)
        else:
            t = random.expovariate(1.0 / INTERWAL_SZCZYT)
        yield env.timeout(t)

        zrodlo_wartosci = zrodlo_wartosci.append({"start_czas": env.now, "numer_w_kolejce": numer_w_kolejce},
                                                 ignore_index=True)


def klient(id_klienta, env, WARTOSC_ZAKUPOW_SR, WARTOSC_ZAKUPOW_ODCH,
           kasa, numer_w_kolejce, WSP_ZWROT, K, LAMBDA, UTRACONY_POTENCJAL):
    start_czekania = env.now
    global klient_wartosci
    print('%s przybył do sklepu o %.1f' % (id_klienta, start_czekania))
    with kasa.request() as req:
        yield req

        oczekiwanie = env.now - start_czekania
        # We got to the counter
        print('%7.4f %s: Czekał w kolejce %6.3f' % (env.now, id_klienta, oczekiwanie))

        prwd_churn = scipy.stats.weibull_min.cdf(oczekiwanie, c=K, scale=LAMBDA)
        churn = np.random.binomial(n=1, p=prwd_churn)
        if churn == 1:
            print('%7.4f %s: Klient obniży lojalność wobec marki. Oczekiwana utrata przychodu: %6.0f' % (
            env.now, id_klienta, UTRACONY_POTENCJAL))

        wartosc_zakupow = random.normalvariate(WARTOSC_ZAKUPOW_SR, WARTOSC_ZAKUPOW_ODCH)
        print(('%7.4f %s zrobił zakupy na kwotę: %.1f' % (env.now, id_klienta, wartosc_zakupow)))

        # przyznanie zwrotu
        if numer_w_kolejce > 5:
            zwrot = WSP_ZWROT * wartosc_zakupow
            print(('%7.4f %s otrzymal zwrot: %.1f' % (env.now, id_klienta, zwrot)))
        else:
            zwrot = 0
            print(('%7.4f %s nie otrzymał zwrotu' % (env.now, id_klienta)))

        # Kasjerka potrzebuje chwili
        czas_obslugi = ST_CZAS_OBSLUGI + (wartosc_zakupow * SZYBKOSC_OBSLUGI)
        yield env.timeout(czas_obslugi)

        global klienci_obsluzeni
        klienci_obsluzeni += 1

        print('%s został obsłużony %.1f sekund.' % (id_klienta,
                                                    env.now - start_czekania))

        klient_wartosci = klient_wartosci.append({'id_klienta': id_klienta,
                                                  "oczekiwanie": oczekiwanie,
                                                  "wartosc_zakupow": wartosc_zakupow,
                                                  "zwrot": zwrot,
                                                  "czas_obslugi": czas_obslugi,
                                                  "capacity": np.NaN,
                                                  "utracony_potencjal": churn * UTRACONY_POTENCJAL}, ignore_index=True
                                                 )


print('Symulacja kolejki')
random.seed(RANDOM_SEED)
env = simpy.Environment()
ultimate_result = pd.DataFrame(columns=['id_klienta', 'oczekiwanie', 'wartosc_zakupow', 
                                        'zwrot', 'czas_obslugi', 'utracony_potencjal'])
for i in range(10):
    for n in range(1, 11, 1):
        # Uruchomienie symulacji
        kasa = simpy.Resource(env, capacity=n)
        env.process(zrodlo(env, liczba_klientow, INTERWAL_SPOKOJ, INTERWAL_SZCZYT, kasa, klienci_obsluzeni))
        env.run()
        klient_wartosci['capacity'] = klient_wartosci['capacity'].fillna(n)

        # Resetowanie zmiennych
        klienci_obsluzeni = 0
        numer_w_kolejce = 0
        liczba_klientow = round(random.normalvariate(SR_LICZBA_KLIENTOW, ODCH_LICZBA_KLIENTOW))

    result = pd.concat([klient_wartosci, zrodlo_wartosci], axis=1)
    r1 = result.iloc[:,0:2].groupby("capacity").count()
    r2 = result.iloc[:,1:7].groupby("capacity").mean()
    r1r2 = r1.merge(r2, left_index = True, right_index = True)
    ultimate_result = ultimate_result.append(r1r2)
    
    klient_wartosci = pd.DataFrame(columns=['id_klienta', 'capacity', 'oczekiwanie', 
                                             'wartosc_zakupow', 'zwrot', 'czas_obslugi',
                                             "utracony_potencjal"], dtype=np.float64)

    zrodlo_wartosci = pd.DataFrame(columns=['start_czas', "numer_w_kolejce"], dtype=np.float64)

ultimate_result['id_klienta'] = ultimate_result['id_klienta'].astype(np.int64)    
ultimate_result.reset_index(drop=False, inplace=True)
ultimate_result.rename(columns = {"index": "capacity"}, inplace = True)
    

s1 = ultimate_result.iloc[:,0:2].groupby("capacity").mean()
s2 = ultimate_result.groupby("capacity").mean()
summary = s1.merge(s2, left_index = True, right_index = True)
summary.reset_index(drop=False, inplace=True)
summary.rename(columns = {"index": "capacity"}, inplace = True)









