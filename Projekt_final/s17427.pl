% Patryk Parulski
% s17427

%Pomieszczenia, przedmioty and koty
%===========================================================

% mieszkanie
mieszkanie.
jeans.
sztruks.
tshirt.
koszula.
szafa(['jeans','sztruks','tshirt','koszula']).

% pomieszczenia 
pomieszczenie(przedpokoj).
pomieszczenie(sypialnia).
pomieszczenie(wc).
pomieszczenie(lazienka).
pomieszczenie(salon).
pomieszczenie(kuchnia).
pomieszczenie(schowek).

zwierze(kot_1).
zwierze(kot_2).

% dodanie pomieszczen do mieszkania
pomieszczenie_w_mieszkaniu(przedpokoj,mieszkanie).
pomieszczenie_w_mieszkaniu(sypialnia,mieszkanie).
pomieszczenie_w_mieszkaniu(wc,mieszkanie).
pomieszczenie_w_mieszkaniu(lazienka,mieszkanie).
pomieszczenie_w_mieszkaniu(salon,mieszkanie).
pomieszczenie_w_mieszkaniu(kuchnia,mieszkanie).
pomieszczenie_w_mieszkaniu(schowek,mieszkanie).

idz(przedpokoj,salon).
idz(przedpokoj,kuchnia).
idz(przedpokoj,lazienka).
idz(przedpokoj,wc).
idz(przedpokoj,sypialnia).
idz(salon,schowek).
spowrotem(X,Y) :- idz(Y,X).

% def
jest_w(szafka_na_buty,przedpokoj).
jest_w(drapak,salon).
jest_w(tv,salon).
jest_w(stolik_kawowy,salon).
jest_w(stol_jadalny,salon).
jest_w(krzeslo_1,salon).
jest_w(krzeslo_2,salon).
jest_w(krzeslo_3,salon).
jest_w(krzeslo_4,salon).
jest_w(kanapa,salon).
jest_w(lustro,lazienka).
jest_w(wanna,lazienka).
jest_w(prysznic,lazienka).
jest_w(szafka_lazienkowa,lazienka).
jest_w(umywalka_3,lazienka).
jest_w(umywalka_2,wc).
jest_w(sedes,wc).
jest_w(kot_1,salon).
jest_w(lozko,sypialnia).
jest_w(szafa,sypialnia).
jest_w(lodowka,kuchnia).
jest_w(polka,kuchnia).
jest_w(blat,kuchnia).
jest_w(umywalka_1,kuchnia).
jest_w(kot_2,kuchnia).
jest_w(skarb,schowek).

miedzy(stolik_kawowy,(krzeslo_4,tv)).
miedzy(stolik_kawowy,(krzeslo_3,tv)).
miedzy(stolik_kawowy,(drapak,kanapa)).
miedzy(stol_jadalny,(krzeslo_1,krzeslo_4)).
miedzy(stol_jadalny,(krzeslo_2,krzeslo_3)).

miedzy(umywalka_3,(lustro,wanna)).
miedzy(umywalka_3,(lustro,prysznic)).
miedzy(umywalka_3,(lustro,szafka_lazienkowa)).
miedzy(szafka_lazienkowa,(prysznic, wanna)).


na(kot_1,drapak).
na(kot_2,blat).
na(umywalka_1,blat).


schowane_w(kasa^za^wybory,skarb).


%Dzialania
%===========================================================
:- dynamic obecne_pomieszczenie/1.
:- dynamic poglaskane_kotki/1.

w_mieszkaniu(X,Y) :- pomieszczenie_w_mieszkaniu(Z,Y), jest_w(X, Z).

rzeczy_w_mieszkaniu(X,Y) :- pomieszczenie_w_mieszkaniu(Z,Y), jest_w(X, Z), \+ zwierze(X).

co_robi_kot(X) :- zwierze(X),(   
                              na(X, drapak) -> write(X),write(' bawi się'),nl;
                              na(X, blat) -> write(X),write(' psoci'),nl;
                              write(X),write(' nie wiadomo, pewnie psuje'),nl
                              ).
                            

czy_jest_w_szafie(X) :- obecne_pomieszczenie(Y), jest_w(szafa, Z), (
                                            Y \= Z -> write('Idz do sypialni by to sprawdzic'),nl;
                                            Y == Z -> (szafa(V), member(X, V))
).

ile_rzeczy_w_szafie :- obecne_pomieszczenie(Y), jest_w(szafa, Z), (
                                            Y \= Z -> write('Idz do sypialni by to sprawdzic'),nl;
                                            Y == Z -> (
                                                szafa(L), length(L,X), write('szafa ma '), write(X), write(' rzeczy'), nl
                                            )
).

gdzie_moge_isc_z(X) :- (idz(X,Y);spowrotem(X,Y)),format('Z ~w mozesz isc do ~w',[X,Y]).
gdzie_moge_isc :- obecne_pomieszczenie(X),(idz(X,Y);spowrotem(X,Y)),format('Z ~w mozesz isc do ~w',[X,Y]).

przejdz_do(X) :- obecne_pomieszczenie(Y), (
                                            X == Y -> write('Juz tutaj jestes'),nl;
                                            (idz(Y,X);spowrotem(Y,X)) -> 
                                                write('Jestes w: '),write(X),nl,retract(obecne_pomieszczenie(Y)), assert(obecne_pomieszczenie(X));
                                            \+ (idz(Y,X);spowrotem(Y,X)) -> 
                                                write('Nie mozna przejsc bezposrednio do: '),write(X),nl,retract(obecne_pomieszczenie(Y)), assert(obecne_pomieszczenie(X));
                                                fail
).


gdzie_kot(X) :- na(X,Y), format('~w bedacego na ~w',[X,Y]).

poglaszcz_kotka(X) :- obecne_pomieszczenie(Y), zwierze(X), (
    (jest_w(X,Z),Y\=Z) -> write('Nie mozna - kotek jest w: '),write(Z),nl;
    (jest_w(X,Z),Y==Z) -> write('Poglaskales: '),gdzie_kot(X),nl,retract(poglaskane_kotki(X)),assert(poglaskane_kotki(X));
    fail
).

co_widze(X) :- obecne_pomieszczenie(Y), jest_w(X,Y).

otworz_skarb :- co_widze(Y), jest_w(Z, schowek), schowane_w(X, Z), Y==Z, format('Odnalazles ~w ',[X]).

start :- write('Jestes w przedpokoju'),nl,
         write('Sprawdz ile rzeczy jest w szafie w sypialni i czy znajduje sie w niej tshirt'),nl,
         write('Znajdz koty, poglaszcz kot_1 oraz kot_2.'),nl,
         write('Odszukaj skarb :)'),nl,nl,
         komendy, 
         retractall(poglaskane_kotki(_)),
         retractall(obecne_pomieszczenie(_)),
         assert(obecne_pomieszczenie(przedpokoj)).

komendy :- 
         write('By dowiedziec sie, gdzie jestes                    - obecne_pomieszczenie(X).'),nl,
         write('By przejsc do pomieszczenia                        - przejdz_do(nazwa_Pomieszczenia).'),nl,
         write('By sprawdzic gdzie mozna isc z obecnego pom.       - gdzie_moge_isc.'),nl,
         write('By sprawdzic ilosc rzeczy w szafie                 - ile_rzeczy_w_szafie.'),nl,
         write('By sprawdzic, czy cos jest w szafie                - czy_jest_w_szafie(nazwa_Rzeczy).'),nl,
         write('By poglaskac kota                                  - poglaszcz_kotka(nazwa_Kota).'),nl,
         write('By otworzyc skarb                                  - otworz_skarb.'),nl,
         write('By dowiedziec sie jakie sa rzeczy w mieszkaniu     - rzeczy_w_mieszkaniu(X,mieszkanie).'),nl,
         write('By dowiedziec sie co jest w obecnym pomieszczeniu  - co_widze(X).'),nl,
         write('By sprawdzic czy oba kotki zostaly poglaskane      - poglaskane_kotki(X).'),nl,
         write('Aby wyswietlic komendy                             - komendy.').