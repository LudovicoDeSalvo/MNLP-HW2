from transformers import AutoTokenizer

def calculate_token_count(model_name, text_to_check):
    """
    Loads a tokenizer and calculates the number of tokens for a given text.
    """
    try:
        # Load the specific tokenizer for the model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Encode the text to get the list of token IDs
        token_ids = tokenizer.encode(text_to_check)
        
        # The number of tokens is the length of the list of IDs
        token_count = len(token_ids)
        
        print(f"Model: {model_name}")
        print(f"The text has {token_count} tokens.")
        
        # Check against the typical context limit
        if token_count > 2048:
            print(f"Warning: This text exceeds the typical 2048 token limit for this model.")
        else:
            print("This text is within the typical 2048 token limit.")
            
        return token_count

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# --- Main part of the script ---
if __name__ == '__main__':
    # Define the model you are checking against
    # You can change this to "sapienzanlp/Minerva-350M-base-v1.0" to compare
    TARGET_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Paste the paragraph you want to analyze here
    my_paragraph = """
XXXV. 
Pinocchio  ritrova  in  corpo  al  Pesce-cane....   chi  ritrova? 
Leggete  questo  capitolo  e  lo   saprete. 
Pinoccliio,  appena  che  ebbe  detto  addio  al 
suo  buon  amico  Tonno,  si  mosse  brancolando  in 
mezzo  a  quel  buio,  e  camminando  a  tastoni  den- 
tro il  corpo  del  Pesce-cane,  si  avviò,'  un  passo 
dietro  r  altro,  verso  quel  piccolo  chiarore  che  ve- 
deva baluginare  lontano  lontano. 
E  nel  camminare  sentì  che  i  suoi  piedi  sguazza- 
vano in  una  pozzanghera  d'acqua  grassa  e  sdruc- 
ciolona,  e  quell'acqua  sapeva  di  un  odore  così 
acuto  di  pesce  fritto,  che  gli  pareva  d'essere  a 
mezza  quaresima. 
E  più  andava  avanti,  e  più  il  chiarore  si  faceva 
rilucente  e  distinto:  finché,  cammina  cammina, 
alla  fine  arrivò  :  e  quando  fu  arrivato....  che  cosa 
trovò?  Ve  lo  do  a  indovinare  in  mille:  trovò  una 
piccola  tavola  apparecchiata,  con  sopra  una  can- 
dela accesa  infilata  in  una  bottiglia  di  cristallo 
verde,  e  seduto  a  tavola  un  vecchiettino  tutto 


bianco,  come  se  fosse  di  neve  o  di  panna  montata, 
il  qnale  se  ne  stava  lì  biascicando  alcuni  pescio- 
lini vivi,  ma  tanto  vivi,  che  alle  volte,  mentre  li 
mangiava,  gli  scappavano  perfino  di  bocca. 


E  più  andava  avanti,  e  più  il  chiarore  si  faceva  rilucente. 

A  quella  vista  il  povero  Pinoccliio  ebbe  un'al- 
legrezza così  grande  e  così  inaspettata,  che  ci 
mancò  un  ètte  che  non  cadesse  in  delirio.  Voleva 
ridere,  voleva  piangere,  voleva  dire  un  monte  di 
cose;  e  invece  mugolava  contusamente  e  balbet- 
tava delle  i)arole  tronche  e  sconclusionate.  Final- 


mente  gli  riuscì  di  cacciar  fuori  un  grido  di  gioia, 
e  spalancando  le  braccia  e  gettandosi  al  collo  del 
vecchietto,  cominciò  a  urlare: 
—  Oh!  babbino  mio!  finalmente  vi  ho  ritrovato! 
Ora  poi  non  vi  lascio  più,  mai  più,  mai  più! 


Gettandosi  al  collo  del  vecchietto,  cominciò  a  urlare. 

—  Dunque  gli  occhi  mi  dicono  il  vero  ì  —  re- 
plicò il  vecchietto,  stropicciandosi  gli  occhi.  — 
Dunque  tu  se' proprio  il  mi' caro  Pinocchio! 
—  Sì,  sì!  sono  io,  proprio  io!  E  voi  mi  aA^ete 
digià  perdonato,  non  è  vero?  Oh  babbino  mio, 
come  siete  buono  !...  e  pensare  che  io,  invece....  Oh  ! 
ma  se  sapeste  quante  disgrazie  mi  son  piovute  sul 
capo  e  quante  cose  mi  sono  andate  a  traverso  !  Fi- 


guratevi  che  il  giorno  che  voi,  povero  babbino, 
col  vendere  la  vostra  casacca,  mi  compraste  l' Ab- 
becedario per  andare  a  scuola,  io  scappai  a  ve- 
dere i  burattini,  e  il  burattinaio  mi  voleva  met- 
tere sul  fuoco  i^erchè  gli  cocessi  il  montone  arro- 
sto, che  fu  quello  poi  che  mi  dette  cinque  monete 
d'oro,  perchè  le  portassi  a  voi,  ma  io  trovai  la 
Volpe  e  il  Gatto,  che  mi  condussero  all' Osteria 
del  Gambero  Eosso,  dove  mangiarono  come  lupi, 
e  partito  solo  di  notte  incontrai  gli  assassini  che 
si  messero  a  corrermi  dietro,  e  io  via,  e  loro  die- 
tro, e  io  via,  e  loro  sempre  dietro,  e  io  via,  finche 
m'impiccarono  a  un  ramo  della  Quercia  Grande, 
dovecchè  la  bella  Bambina  dai  capelli  turchini  mi 
mandò  a  prendere  con  una  carrozzina,  e  i  medici, 
quando  m'ebbero  visitato,  dissero  subito:  «  Se 
non  è  morto,  è  segno  che  è  scDipre  vivo  »  e  allora 
mi  scappò  detta  uua  bugia,  e  il  naso  cominciò  a 
crescermi  e  non  mi  passava  più  dalla  porta  di  ca- 
mera, motivo  i)er  cui  andai  con  la  Volpe  e  col 
Gatto  a  sotterrare  le  quattro  monete  d' oro,  che 
una  l' avevo  spesa  all'  Osteria,  e  il  pappagallo  si 
messe  a  ridere,  e  viceversa  di  duemila  monete 
non  trovai  più  nulla,  la  quale  il  Giudice  quando 
seppe  che  ero  stato  derubato,  mi  fece  subito  met- 
tere in  prigione,  per  dare  una  soddisfazione  ai 
ladri,  di  dove,  col  venir  via,  vidi  un  bel  grappolo 


d'uva  in  un  campo,  che  rimasi  preso  alla  tagliola 
e  il  contadino  di  santa  ragione  mi  messe  il  col- 
lare da  cane  perchè  facessi  la  guardia  al  pollaio, 
che  riconobbe  la  mia  innocenza  e  mi  lasciò  an- 
dare, e  il  serpente,  colla  coda  che  gli  fumava,  co- 
minciò a  ridere  e  gli  si  strappò  una  vena  sul  petto, 
e  cosi  ritornai  alla  casa  della  bella  Bambina,  che 
era  morta,  e  il  Colombo  vedendo  che  piangevo  mi 
disse:  «  Ho  visto  il  tu' babbo  che  si  fabbricava 
una  barchettina  per  venirti  a  cercare  »  e  io  gli 
dissi  :  «  Oh  !  se  avessi  le  ali  anch'  io  »  e  lui  mi  disse: 
«  Vuoi  venire  dal  tuo  babbo!  »  e  io  gli  dissi:  «  Ma- 
gari! ma  chi  mi  ci  porta!  »  e  lui  mi  disse.:  «  Ti  ci 
porto  io  »  e  io  gli  dissi:  «  Come!  »  e  lui  mi  disse: 
«  Montami  sulla  groppa  »  e  così  abbiamo  volato 
tutta  la  notte,  poi  la  mattina  tutti  i  pescatori  che 
guardavano  verso  il  mare  mi  dissero:  «  C'è  un 
pover'omo  in  una  barchetta  che  sta  per  affogare  » 
e  io  da  lontano  vi  riconobbi  subito,  perchè  me  lo 
diceva  il  core,  e  vi  feci  segno  di  tornare  alla 
spiaggia.... 
—  Ti  riconobbi  anch'  io,  —  disse  Geppetto  —  e 
sarei  volentieri  tornato  alla  spiaggia:  ma  come 
fare  !  il  mare  era  grosso  e  un  cavallone  m' arro- 
vesciò la  barchetta.  Allora  un  orribile  Pesce-cane 
che  era  lì  vicino,  appena  che  m' ebbe  visto  nel- 
r  acqua,  corse  subito  verso  di  me,  e  tirata  fuori  la 

lingiin,  mi  prese  pari  pari,  e  m'inghiottì  come  im 
tortellino  di  Bologna. 
—  E  qnant'  è  che  siete  rinchiuso  qui  dentro  ?  — 
domandò  Pinocchio. 
—  Da  quel  giorno  in  poi,  saranno  ormai  due 
anni:  due  anni,  Pinocchio  mio....  che  mi  son  parsi 
due  secoli  ! 
—  E  come  avete  fatto  a  campare?  E  dove  avete 
trovatala  candela?  E  i  fiammiferi  per  accenderla, 
chi  ve  li  ha  dati  ? 
—  Ora  ti  racconterò  tutto.  Devi  dunque  sapere 
che  quella  medesima  burrasca,  che  rovesciò  la 
mia  barchetta,  fece  anche  affondare  un  basti- 
mento mercantile.  I  marinai  si  salvarono  tutti,  ma 
il  bastimento  colò  a  fondo,  e  il  solito  Pesce-cane, 
che  quel  giorno  aveva  un  appetito  eccellente, 
dopo  aver  inghiottito  me,  inghiottì  anche  il  ba- 
stimento.... 
—  Come!  Lo  inghiottì  tutto  in  un  boccone  1... — 
domandò  Pinocchio  maravigliato. 
—  Tutto  in  un  boccone:  e  risi)utò  solamente 
r  albero  maestro,  perchè  gli  era  rimasto  fra  i  denti 
come  una  lisca.  Per  mia  gran  fortuna,  quel  basti- 
mento era  carico  non  solo  di  carne  conservata  in 
cassette  di  stagno,  ma  di  biscotto,  ossia  di  pane 
abbrostolito,  di  bottiglie  di  vino,  d'uva  secca,  di 
cacio,  di  caffè,  di  zucchero,  di  candele  steariche 


e  di  scatole  di  fiammiferi  di  cera.  Con  tutta  que- 
sta grazia  di  Dio  ho  potuto  campare  due  anni: 
ma  oggi  sono  agli  ultimi  sgoccioli:  oggi  nella  di- 
spensa non  e'  è  più  nulla,  e  questa  candela,  che 
vedi  accesa,  è  l'ultima  candela  che  mi  sia  ri- 
masta.... 
—  E  dopo? 
—  E  dopo,  caro  mio,  rimarremo  tutt'e  due  al 
buio. 
—  Allora,  babbino  mio,  —  disse  Pinocchio  — 
non  e'  è  tempo  da  perdere.  Bisogna  pensar  subito 
a  fuggire. 
—  A  fuggirei.,  e  come? 
—  Scappando  dalla  bocca  del  Pesce-cane  e  get- 
tandosi a  nuoto  in  mare. 
—  Tu  parli  bene:  ma  io,  caro  Pinocchio,  non 
so  nuotare! 
—  E  che  importa!...  Voi  mi  monterete  a  caval- 
luccio sulle  spalle,  e  io,  che  sono  un  buon  nuota- 
tore, vi  ijorterò  sano  e  salvo  fino  alla  spiaggia. 
—  Illusioni,  ragazzo  mio!  —  replicò  Geppetto, 
SCO  tendo  il  capo  e  sorridendo  malinconicamente. 
—  Ti  pare  egli  possibile  che  un  burattino,  alto 
appena  un  metro  come  sei  tu,  possa  aver  tanta 
forza  da  portarmi  a  nuoto  sulle  spalle? 
—  Provatevi  e  vedrete!  A  ogni  modo,  se  sarà 
scritto  in  cielo  che  dobbiamo  morire,  avremo  al- 


meno  la  gran  coiisolazione  di  morire  abbracciati 
iusieme.  — 
E  senza  dir  altro,  Pinocchio  prese  in  mano  la 
candela,  e  andando  avanti  per  far  lume,  disse 
al  suo  babbo: 
—  Venite  dietro  a  me,  e  non  abbiate  paura.  — 
E  così  camminarono  un  bel  pezzo,  e  traversa- 
rono tutto  il  corpo  e  tutto  lo  stomaco  del  Pesce- 
cane. Ma  giunti  al  x)unto  dove  cominciava  la  spa- 
ziosa gola  del  mostro,  pensarono  bene  di  fermarsi 
per  dare  un'occhiata  e  cogliere  il  momento  op- 
portuno alla  fuga. 
Ora  bisogua  sapere  che  il  Pesce-cane,  essendo 
molto  vecchio  e  soffrendo  d'asma  e  di  palpitazione 
di  cuore,  era  costretto  a  dormire  a  bocca  aperta  : 
per  ciii  Pinocchio  afitac^iandosi  al  principio  della 
gola,  e  guardando  in  su,  potò  vedere  al  di  fuori 
di  quell'enorme  bocca  spalancata  un  bel  pezzo 
di  cielo  stellato  e  un  bellissimo  lume  di  luna. 
—  Questo  è  il  vero  momento  di  scappare  — 
bisbigliò  allora,  voltandosi  al  suo  babbo.  —  Il  Pe- 
sce-cane dorme  come  un  ghiro:  il  mare  è  tran- 
quillo e  ci  si  vede  come  di  giorno.  Venite  dunque, 
babbino,  dietro  a  me,  e  fra  poco  saremo  salvi.  — 
Detto  fatto  salirono  su  per  la  gola  del  mostro 
marino,  e  arrivati  in  quelP  immensa  bocca  comin- 
ciarono a  camminare  in  punta  di  piedi  sulla  lin- 


glia;  una  liugua  così  larga  e  così  Inng^,  che  pa- 
reva il  viottolone  d'un  giardino.  E  già  stavano  lì 
lì  per  fare  il  gran  salto  e  per  gettarsi  a  nuoto  nel 
mare,  quando,  sul  più  bello,  il  Pesce-cane  star- 
nuti, e  nello  starnutire,  détte  uno  scossone  così 
violento,  elle  PinoccMo  e  Geppetto  si  trovarono 
rimbalzati  all'  indietro  e  scaraventati  nuovamente 
in  fondo  allo  stomaco  del  mostro. 
ISTel  grand'  urto  della  caduta  la  candela  si  spen- 
se, e  padre  e  figliuolo  rimasero  al  buio. 
—  E  ora?...  —  domandò  Pinocchio  facendosi 
serio. 
—  Ora,  ragazzo  mio,  siamo  beli' e  perduti. 
—  Perchè  perduti?  Datemi  la  mano,  babbino, 
e  badate  di  non  sdrucciolare!... 
—  Dove  mi  conduci? 
—  Dobbiamo  ritentare  la  fuga.  Venite  con  me 
e  non  abbiate  jjaura.  — 
Ciò  detto,  Pinocchio  prese  il  suo  babbo  per  la 
mano:  e  camminando  sempre  in  punta  di  piedi, 
risalirono  insieme  su  per  la  gola  del  mostro:  poi 
traversarono  tutta  la  lingua  e  scavalcarono  i  tre 
filari  di  denti.  Prima  però  di  fare  il  gran  salto,  il 
burattino  disse  al  suo  babbo: 
—  Montatemi  a  cavalluccio  sulle  spalle  e  ab- 
bracciatemi forte  forte.  Al  resto  ci  penso  io.  — 
Appena  Geijpetto  si  fu  accomodato  per  bene 

tói  gettò  neir acqua  e  comiuciu  a  luiutart;. 


sulle  spalle  del  figliuolo,  il  bravo  Pinocchio,  si- 
curo del  fatto  suo,  si  gettò  nell'acqua  e  cominciò 
a  nuotare.  Il  mare  era  tranquillo  come  un  olio: 
la  luna  splendeva  in  tutto  il  suo  chiarore,  e  il 
Pesce-cane  seguitava  a  dormire  di  un  sonno  così 
profondo,  che  non  l'avrebbe  svegliato  nemmeno 
una  cannonata.  )
    """

    print("-" * 50)
    calculate_token_count(TARGET_MODEL, my_paragraph)
    print("-" * 50)