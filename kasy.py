'''
Market Marek i Mariusz ma kasy, do któych co X (rozkład) minut przychodzą klienci (jak uwzględnić godziny szczytu)
Jeśli przed klientem do kasy stoi więcej niż 5 klient otrzymuje on 10% zwrot kwoty zakupów. Jeśli klient stoi w kasie
za długo może dojśc do zjawiska zrezygnowania z przyszłych zakupów (jakaś dystrybuanta). Wartość zakupu ma pewien
rozkład, a w wypadku churnu tracimy klienta na cały horyzont symulacji, gdzie średnia liczba zkaupów w miesiącu i ich wartość
również będzie losowana wg rozkładu
'''
import random
import simpy
import scipy.stats
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns

#Parametry
ST_CZAS_OBSLUGI = 15    #stały element czasu obsługi - drukowanie paragonu, płatność itd.
SZYBKOSC_OBSLUGI = 1 / 3   # [s/PLN] - czas obsługi zależny od wielkości zakupów - kasowanie kolejnych produktów
WARTOSC_ZAKUPOW_SR = 125 # Średnia wartość zakupów (rozkład normalny)
WARTOSC_ZAKUPOW_ODCH = 30 # Odchylenie średniej wartości zakupów
INTERWAL_SPOKOJ = 100 # Przeciętny interwał pojawiania się kolejnych klientó przy kasie w "spokojnych" godzinach
INTERWAL_SZCZYT = 15 # Przeciętny interwał w godzinach szczytu
SR_LICZBA_KLIENTOW = 500 # Śr dzienna liczba klientów
ODCH_LICZBA_KLIENTOW = 25 # Odchylenie (rozkład nrmalny)
WSP_ZWROT = 0.10 # % zwrotu udzielany dla klientó, którzy mają miejsce w kolejce późniejsze niż 5
RANDOM_SEED = 42


MAX_LICZBA_KAS = 4 # max. rozważana liczba kas
ITERACJE = 50 # liczba iteracji

# parametry powiązane z churn
WSP_UTR_PTN = 0.05  # strata ok. 10% CLV w danym horyzoncie w przypadku churn
PRZ_LICZBA_WIZYT_ROK = 21   # liczba wizyt do kalkulacji utraconego potencjalu na skutek churn
HORYZONT = 1 # Horyzont do obliczeń utraconego potencjału - 1 rok

UTRACONY_POTENCJAL = 0.05 * PRZ_LICZBA_WIZYT_ROK * HORYZONT * WARTOSC_ZAKUPOW_SR
# parametry rozkładu Weibulla
LAMBDA = 210 # parametr lambda - 63% klientów zdecyduje się na churn w przyszłości po 3,5 minutach oczekiwania
K = 3  # wraz ze wzrostem czasu oczekiwania wzrasta p-pstwo rezygnacji klienta z następnych zakupów

klienci_obsluzeni = 0
numer_w_kolejce = 0

liczba_klientow = round(random.normalvariate(SR_LICZBA_KLIENTOW, ODCH_LICZBA_KLIENTOW))

klient_wartosci = pd.DataFrame(
    columns=['id_klienta', 'capacity', 'oczekiwanie', 'wartosc_zakupow', 'zwrot', 'czas_obslugi',
             "utracony_potencjal"], dtype=np.float64)

zrodlo_wartosci = pd.DataFrame(columns=['start_czas', "numer_w_kolejce"], dtype=np.float64)


### GENERATOR KLIENTÓW
def zrodlo(env, liczba_klientow, INTERWAL_SZCZYT, INTERWAL_SPOKOJ, WARTOSC_ZAKUPOW_SR, WARTOSC_ZAKUPOW_ODCH, kasa,
                   klienci_obsluzeni_b, WSP_ZWROT, K, LAMBDA, UTRACONY_POTENCJAL):
    """Przybywanie nowych klientów do kas"""
    global numer_w_kolejce
    global zrodlo_wartosci

    for i in range(liczba_klientow):
        numer_w_kolejce = i - klienci_obsluzeni

        if i < 0.2 * liczba_klientow or i > 0.7 * liczba_klientow:
            t = random.expovariate(1.0 / INTERWAL_SZCZYT)
        else:
            t = random.expovariate(1.0 / INTERWAL_SPOKOJ)
        yield env.timeout(t)
        env.process(klient('Klient%02d' % i, env, WARTOSC_ZAKUPOW_SR, WARTOSC_ZAKUPOW_ODCH,
           kasa, numer_w_kolejce, WSP_ZWROT, K, LAMBDA, UTRACONY_POTENCJAL))
        zrodlo_wartosci = zrodlo_wartosci.append({"start_czas": env.now, "numer_w_kolejce": numer_w_kolejce},
                                                 ignore_index=True)


def klient(id_klienta, env, WARTOSC_ZAKUPOW_SR, WARTOSC_ZAKUPOW_ODCH,
           kasa, numer_w_kolejce, WSP_ZWROT, K, LAMBDA, UTRACONY_POTENCJAL):

    start_czekania = env.now
    global klient_wartosci
    print('%s przybył do kas o %.1f' % (id_klienta, start_czekania))
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
        if wartosc_zakupow <= 0:
            wartosc_zakupow = 0.20
        print(('%7.4f %s zrobił zakupy na kwotę: %.1f' % (env.now, id_klienta, wartosc_zakupow)))

        # przyznanie zwrotu
        if numer_w_kolejce > 5:
            zwrot = WSP_ZWROT * wartosc_zakupow
            print(('%7.4f %s miał numer kolejce: %3.0f i otrzymal zwrot: %.1f' % (env.now, id_klienta, numer_w_kolejce, zwrot)))
        else:
            zwrot = 0
            print(('%7.4f %s miał numer w kolejce %3.0f i nie otrzymał zwrotu' % (env.now, id_klienta, numer_w_kolejce)))

        # Kasjerka potrzebuje chwili
        czas_obslugi = ST_CZAS_OBSLUGI + (wartosc_zakupow * SZYBKOSC_OBSLUGI)
        yield env.timeout(czas_obslugi)

        global klienci_obsluzeni
        klienci_obsluzeni = klienci_obsluzeni + 1

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

'''Uruchomienie symulacji'''
# Inicjalizacja srodowiska oraz utworzenie DF przechowującego skumulowane wyniki dla kazdej iteracji oraz liczby kas
random.seed(RANDOM_SEED)
env = simpy.Environment()
# ultimate_result będzie przechowywał wyniki dla kazdej iteracji z podzialem na capacity
# W pozniejszym etapie tworzony jest DF summary, ktory dodatkowo tworzy srednie dla iteracji
ultimate_result = pd.DataFrame(columns=['id_klienta', 'oczekiwanie', 'wartosc_zakupow',
                                        'zwrot', 'czas_obslugi', 'utracony_potencjal',
                                        'SZYBKOSC_OBSLUGI', 'INTERWAL_SZCZYT', 'SR_LICZBA_KLIENTOW'])
    
# Analiza wrazliwosci - tworzenie badanego zakresu zmiennych oraz grid wykorzystywany do iterowania
a_szybkosc_obslugi = np.array(np.arange(0.25,1.01,0.25))
a_interwal_szczyt = np.array(range(15,100,15))
a_sr_liczba_klientow = np.array(range(300,701,100))

# Uposzczony grid (jedna ruchoma zmienna, reszta stała)
a_szybkosc_obslugi_moving = pd.DataFrame(data={"SZYBKOSC_OBSLUGI":a_szybkosc_obslugi,
                                               "INTERWAL_SZCZYT": INTERWAL_SZCZYT,
                                               "SR_LICZBA_KLIENTOW": SR_LICZBA_KLIENTOW})
a_szybkosc_obslugi_moving = a_szybkosc_obslugi_moving.fillna(method="ffill")

a_interwal_szczyt_moving = pd.DataFrame(data={"SZYBKOSC_OBSLUGI":SZYBKOSC_OBSLUGI,
                                               "INTERWAL_SZCZYT": a_interwal_szczyt,
                                               "SR_LICZBA_KLIENTOW": SR_LICZBA_KLIENTOW})
a_interwal_szczyt_moving = a_interwal_szczyt_moving.fillna(method="ffill")

a_sr_liczba_klientow_moving = pd.DataFrame(data={"SZYBKOSC_OBSLUGI":SZYBKOSC_OBSLUGI,
                                               "INTERWAL_SZCZYT": INTERWAL_SZCZYT,
                                               "SR_LICZBA_KLIENTOW": a_sr_liczba_klientow})
a_sr_liczba_klientow_moving = a_sr_liczba_klientow_moving.fillna(method="ffill")

grid_ready = pd.concat([a_szybkosc_obslugi_moving, a_interwal_szczyt_moving, a_sr_liczba_klientow_moving])

# Grid tworzony jako wszyskie możliwe kombinacje - zdecydowanie dłuższy czas wykonania
# grid = np.meshgrid(a_szybkosc_obslugi, a_interwal_szczyt, a_sr_liczba_klientow)
# grid_ready = pd.DataFrame(data={"a_szybkosc_obslugi": np.ravel(grid[0]),
#                                 "a_interwal_szczyt": np.ravel(grid[1]),
#                                 "a_sr_liczba_klientow": np.ravel(grid[2])})

# Uruchomienie symulacji - rozne kombinacje zmiennych oraz liczby kas - kazdy uklad badany n iteracji
for row in range(grid_ready.shape[0]):
    SZYBKOSC_OBSLUGI = float(grid_ready.iloc[row,0])
    INTERWAL_SZCZYT = int(grid_ready.iloc[row,1])
    SR_LICZBA_KLIENTOW = int(grid_ready.iloc[row,2])
    for i in range(ITERACJE):
        for n in range(1, MAX_LICZBA_KAS + 1, 1):
            # Uruchomienie symulacji
            kasa = simpy.Resource(env, capacity=n)
            env.process(zrodlo(env, liczba_klientow, INTERWAL_SZCZYT, INTERWAL_SPOKOJ, WARTOSC_ZAKUPOW_SR, WARTOSC_ZAKUPOW_ODCH, kasa,
                       klienci_obsluzeni, WSP_ZWROT, K, LAMBDA, UTRACONY_POTENCJAL))
            env.run()
            klient_wartosci['capacity'] = klient_wartosci['capacity'].fillna(n)
    
            # Resetowanie zmiennych
            klienci_obsluzeni = 0
            numer_w_kolejce = 0
            liczba_klientow = round(random.normalvariate(SR_LICZBA_KLIENTOW, ODCH_LICZBA_KLIENTOW))
        
        # Grupwanie wynikow dla kazdego capacity 
        result = pd.concat([klient_wartosci, zrodlo_wartosci], axis=1, sort=True)
        r1 = result.iloc[:, 0:2].groupby("capacity").count()
        r2 = result.loc[:, ['capacity', 'utracony_potencjal', 'zwrot']].groupby("capacity").sum()
        r3 = result.loc[:, ['capacity', 'oczekiwanie', 'wartosc_zakupow', "czas_obslugi"]].groupby("capacity").mean()
        r1r2r3 = pd.concat([r1, r2, r3], axis=1, sort=True)
        ultimate_result = ultimate_result.append(r1r2r3, sort=True)
        ultimate_result = ultimate_result.fillna({"SZYBKOSC_OBSLUGI": grid_ready.iloc[row,0], 
                                                  "INTERWAL_SZCZYT": grid_ready.iloc[row,1],
                                                  "SR_LICZBA_KLIENTOW": grid_ready.iloc[row,2]})
    
        klient_wartosci = pd.DataFrame(columns=['id_klienta', 'capacity', 'oczekiwanie',
                                                 'wartosc_zakupow', 'zwrot', 'czas_obslugi',
                                                 "utracony_potencjal"], dtype=np.float64)

        zrodlo_wartosci = pd.DataFrame(columns=['start_czas', "numer_w_kolejce"], dtype=np.float64)

ultimate_result['id_klienta'] = ultimate_result['id_klienta'].astype(np.int64)
ultimate_result.reset_index(drop=False, inplace=True)
ultimate_result.rename(columns = {"index": "capacity"}, inplace = True)

# Grupowanie zbiorczych wyników po capacity - srednie wyniki dla wszystkich iteracji (oraz kombinacji 
# jeżeli wykorzystano wariant z analizą wrażliwosci) zaleznie od capacity
s1 = ultimate_result.iloc[:,0:2].groupby("capacity").mean()
s2 = ultimate_result.groupby("capacity").mean()
summary = s1.merge(s2, left_index = True, right_index = True)
summary.reset_index(drop=False, inplace=True)
summary.rename(columns = {"index": "capacity", "czas_obslugi_y": "czas_obslugi"}, inplace = True)
#summary.drop(columns="czas_obslugi_x", inplace=True)

# Sensitivity grouping
h = ultimate_result.shape[0]
sns1 = ultimate_result.iloc[0:int((h * 1/3) - 1),:].groupby("SZYBKOSC_OBSLUGI").mean()
sns2 = ultimate_result.iloc[int(h * 1/3):int((h * 2/3) - 1),:].groupby("INTERWAL_SZCZYT").mean()
sns3 = ultimate_result.iloc[int(h * 2/3):h-1,:].groupby("SR_LICZBA_KLIENTOW").mean()

writer = pd.ExcelWriter('sns1512.xlsx', engine='xlsxwriter')

sns1.to_excel(writer, sheet_name="SZYBKOSC_OBSLUGI")
sns2.to_excel(writer, sheet_name="INTERWAL_SZCZYT" )
sns3.to_excel(writer, sheet_name="SR_LICZBA_KLIENTOW")

# Close the Pandas Excel writer and output the Excel file.
writer.save()

# Wykresy - analiza wrażliwości
plt.plot(sns1.loc[:,1], sns1.loc[:,6])
plt.plot(sns1.index, sns1.loc[:,['utracony_potencjal']])
plt.plot(sns2.index, sns2.loc[:,['utracony_potencjal']])
plt.plot(sns3.index, sns2.loc[:,['utracony_potencjal']])

plt.show()

# Wykresy

# Ze względu na seryjne tworzenie wykresów warto jest wymusic wyswietlanie w oknie konsoli
# %matplotlib inline

cols = summary.iloc[:,1:].columns

# Wykres dla summary
for i in cols:
    fig, ax = plt.subplots(1,1)
    ax.hist(summary[i], color="blue", edgecolor="black", label=i)
    ax.set_xlabel(i)
    ax.set_ylabel("count")
    ax.set_title("Histogram dla zmiennej {}".format(i))

for i in cols:
    fig, ax = plt.subplots(1,1)
    ax.hist(summary[i], color="blue", edgecolor="black", label=i, cumulative=True, density=True)
    ax.set_xlabel(i)
    ax.set_ylabel("count")
    ax.set_title("CDF dla zmiennej {}".format(i))

for i in cols[[1,3,4,5]]:
    sns.lmplot('capacity', i, data=summary)
    plt.xlabel('capacity')
    plt.ylabel(i)
    plt.title("Zaleznosc {} od {}".format(i,'capacity'))

plt.clf()
df_corr = summary.iloc[:,[0,2,4,5,6]].corr(method='spearman')
sns.heatmap(df_corr)
plt.title("Macierz korelacji")


# Wykresy dla ultimate_results
for i in cols[[1,3,4,5]]:
    fig, ax = plt.subplots(1,1)
    sns.boxplot(x='capacity', y=i, data=ultimate_result, orient='v', ax=ax)
    plt.xlabel('capacity')
    plt.ylabel(i)
    plt.title("Wykres pudełkowy dla {} z podziałem na {}".format(i,'capacity'))

for i in cols[[1,3,4,5]]:
    fig, ax = plt.subplots(1,1)
    sns.swarmplot(x='capacity', y=i, data=ultimate_result, orient='v', ax=ax)
    plt.xlabel('capacity')
    plt.ylabel(i)
    plt.title("Obserwacje dla {} z podziałem na {}".format(i,'capacity'))

for i in cols[[1,3,4,5]]:
    sns.lmplot('oczekiwanie', "utracony_potencjal", data=ultimate_result)
    plt.xlabel('oczekiwanie')
    plt.ylabel("utracony_potencjal")
    plt.title("Zaleznosc oczekiwanie od utracony_potencjal")


plt.show()