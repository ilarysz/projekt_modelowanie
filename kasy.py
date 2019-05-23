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

ST_CZAS_OBSLUGI = 10
SZYBKOSC_OBSLUGI = 1/20
WARTOSC_ZAKUPOW_SR = 80
WARTOSC_ZAKUPOW_ODCH = 20
INTERWAL_SPOKOJ = 1000
INTERWAL_SZCZYT = 150
SR_LICZBA_KLIENTOW = 100
ODCH_LICZBA_KLIENTOW = 10
HORYZONT = 30 # chyba nie będzie potrzebny
RANDOM_SEED = 42
WSP_ZWROT = 0.10

klienci_obsluzeni = 0
numer_w_kolejce = 0

liczba_klientow = round(random.normalvariate(SR_LICZBA_KLIENTOW, ODCH_LICZBA_KLIENTOW))

def zrodlo(env, liczba_klientow, INTERWAL_SPOKOJ, INTERWAL_SZCZYT, kasa, klienci_obsluzeni):
    """Source generates customers randomly"""
    global numer_w_kolejce
    for i in range(liczba_klientow):
        c = klient('Klient%02d' % i, env, WARTOSC_ZAKUPOW_SR, WARTOSC_ZAKUPOW_ODCH, kasa, numer_w_kolejce, WSP_ZWROT)

        numer_w_kolejce = i - klienci_obsluzeni
        env.process(c)
        if env.now < 28800 or env.now > 46800 :
            t = random.expovariate(1.0 / INTERWAL_SPOKOJ)
        else:
            t = random.expovariate(1.0 / INTERWAL_SZCZYT)
        yield env.timeout(t)



def klient(id_klienta, env, WARTOSC_ZAKUPOW_SR, WARTOSC_ZAKUPOW_ODCH, kasa, numer_w_kolejce, WSP_ZWROT):

    start_czekania = env.now
    print('%s przybył do sklepu o %.1f' % (id_klienta, start_czekania))
    with kasa.request() as req:
        yield req

        oczekiwanie = env.now - start_czekania
        # We got to the counter
        print('%7.4f %s: Czekał w kolejce %6.3f' % (env.now, id_klienta, oczekiwanie))


        wartosc_zakupow = random.normalvariate(WARTOSC_ZAKUPOW_SR, WARTOSC_ZAKUPOW_ODCH)
        print(('%7.4f %s zrobił zakupy na kwotę: %.1f' % (env.now, id_klienta, wartosc_zakupow)))

        #przyznanie zwrotu
        if numer_w_kolejce > 5:
            zwrot = WSP_ZWROT*wartosc_zakupow
            print(('%7.4f %s otrzymal zwrot: %.1f' % (env.now, id_klienta, zwrot)))
        else:
            print(('%7.4f %s nie otrzymał zwrotu' % (env.now, id_klienta)))

        # Kasjerka potrzebuje chwili
        czas_obslugi = ST_CZAS_OBSLUGI + (wartosc_zakupow / SZYBKOSC_OBSLUGI)
        yield env.timeout(czas_obslugi)

        global klienci_obsluzeni
        klienci_obsluzeni += 1

        print('%s został obsłużony %.1f sekund.' % (id_klienta,
                                                          env.now - start_czekania))

print('Symulacja kolejki')
random.seed(RANDOM_SEED)
env = simpy.Environment()

kasa = simpy.Resource(env, capacity=1)
env.process(zrodlo(env, liczba_klientow, INTERWAL_SPOKOJ, INTERWAL_SZCZYT, kasa, klienci_obsluzeni))
env.run()