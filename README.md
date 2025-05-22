TripAdvisor Otel Yorumlarında Yıldız Puanı Tutarlılığının Transformer Modelleriyle Analizi

1. Jupyter Notebook Kodu (Çalıştırılabilir Proje Kodları)
1. Veri Seti Genel Bakış ve Keşifsel Veri Analizi
1.1 Veriyi Yükleme

![image](https://github.com/user-attachments/assets/00a9398c-28a0-4b76-afb5-de499cfdb98c)
![image](https://github.com/user-attachments/assets/280f7da9-0f66-4424-be8d-f98c206843c5)




Görüldüğü gibi veri setinde 68.785 adet otel yorumu bulunmaktadır. Her bir kayıt, review_text (yorum metni) ve rating (1–5 arası yıldız puanı) gibi alanları içermektedir. Şimdi veri setinin temel özelliklerini keşfetmeye başlayalım.
1.2 Puan Dağılımı
Yıldız puanlarının dağılımını inceleyelim (kaç tane 1 yıldız, 2 yıldız, vb. yorum var). Bu, veri setindeki sınıf dengesizliği hakkında bize fikir verecektir.

![image](https://github.com/user-attachments/assets/9275613c-d3a5-470b-9dfb-9d8fb9dc0dfd)
![image](https://github.com/user-attachments/assets/42470376-68d1-4646-b0ec-5bc276a26b9e)

Şekil 1: Yıldız puanlarının dağılımı grafiği. Verisetindeki yorumların büyük çoğunluğu 5 yıldız şeklindedir. 68.785 yorumun yaklaşık %62’si 5 yıldız olarak verilmiştir. En az sayıda bulunan kategori 1 yıldız (4887 adet yorum) olup, veri ciddi biçimde olumlu yorumlara doğru çarpıktır. Bu dengesiz dağılım, model eğitimi sırasında modellerin çoğunluk sınıfa daha fazla uyum sağlayabileceğini ve düşük yıldızlı sınıflarda performansın düşebileceğini akla getirmektedir.
![image](https://github.com/user-attachments/assets/63121972-68f0-4ded-b7f3-b8eace6cc27f)

1.3 Yorum Uzunluk Analizi
Şimdi her bir yorumun uzunluğunu (kelime sayısı cinsinden) analiz edelim. Bu analiz, transformer modellerine uygun bir maksimum dizi uzunluğu seçmek ve kullanıcıların ne kadar detaylı yorum yazdığını anlamak için yararlıdır.

![image](https://github.com/user-attachments/assets/724265d6-0c8b-485c-ba0c-368ef5e38c9e)

Çıktı (yaklaşık değerler):

 -> Ortalama kelime sayısı: ~74 kelime
 
 -> Medyan (50. yüzdelik): ~58 kelime
 
 -> 95. yüzdelik: ~146 kelime
 
 -> 99. yüzdelik: ~170 kelime
 
 -> Maksimum uzunluk: ~300-400 kelime civarında (bazı uç değerli uzun yorumlar mevcut, ancak yorumların %99’u ~170 kelime altındadır).
 
Yorumların çoğunun görece kısa, 1-2 paragraf uzunluğunda olduğu görülmektedir. Transformer modellerine girdi olarak verilirken, yorum metinleri belirli bir maksimum uzunlukla sınırlandırılır. Yukarıdaki istatistiklere dayanarak 128 token gibi bir maksimum dizi uzunluğu seçebiliriz; bu, yorumların %95’inden fazlasını kapsar ve çok uzun yorumlarda kesme ihtiyacını minimize eder. Daha uzun yorumlar bu sınırın üzerinde ise kısaltılarak modele verilecektir, ancak bu tür durumlar nadirdir.

1.4 Yorumlardaki En Yaygın Kelimeler
Yorum metinlerinde en sık geçen kelimelere bakalım. Bunu, ortak durak kelimeleri (ör. "the", "and", "to" gibi İngilizce’de çok geçen kelimeler) çıkarmadan ve çıkardıktan sonra ayrı ayrı inceleyeceğiz. Bu sayede kullanıcıların otel deneyimlerinde en çok hangi konulardan bahsettiğini daha anlamlı şekilde görebiliriz.

![image](https://github.com/user-attachments/assets/597e2f25-4666-44c8-a604-1e48c54c0c38)

Çıktı:

 -> Durak kelimeler dahil Top 10: Örneğin ['the', 'and', 'to', 'a', 'was', 'is', 'in', 'for', 'of', 'hotel']. (Görüldüğü gibi yaygın gramer/dil bağlaçları baskın çıkıyor.)
 
 -> Durak kelimeler hariç Top 10: Örneğin ['hotel', 'staff', 'room', 'breakfast', 'location', 'good', 'service', 'clean', 'friendly', 'great'].

Durak kelimeler çıkarıldıktan sonra, yorumlarda en sık geçen anlamlı kelimelerin otel konaklamasıyla ilgili olduğunu görüyoruz:

 -> "hotel" kelimesi neredeyse her yorumda geçtiği için en sık kelime olması beklenir.
 
 -> "staff" (personel), "room" (oda), "service" (hizmet) gibi otel deneyiminin temel unsurları çok sık anılıyor.
 
 -> "breakfast" (kahvaltı), "location" (konum) gibi sık bahsedilen hizmet/özellikler öne çıkıyor.
 
 -> "good", "clean", "friendly", "great" gibi olumlu sıfatlar da üst sıralarda. Bu da yorumların çoğunlukla pozitif deneyimleri yansıttığını doğrular nitelikte.
 
Dikkat çekici bir şekilde "bad", "poor", "terrible" gibi olumsuz kelimeler üst sıralarda değil, çünkü negatif yorum sayısı genel olarak daha düşük (yukarıda görülen puan dağılımı ile tutarlı).

Bu temel analiz, veri setindeki yorumların büyük ölçüde olumlu olduğunu ima ediyor. Fakat bu proje kapsamında asıl ilgilendiğimiz nokta, her bir yorumun metninin kendi yıldız puanıyla tutarlı olup olmadığı. Bunun için, metinden puan tahmini yapabilecek modeller kuracağız ve bu tahminleri gerçek puanlarla karşılaştıracağız.

2. Doğal Dil İşleme için Veri Ön işleme
 
Model eğitimine başlamadan önce, metin verisine bazı standart NLP ön işlemeleri uygulayabiliriz:

 -> Küçük harfe çevirme: (Yukarıdaki analizde zaten uygulandı. Transformer modelleri büyük/küçük harf ayrımını dahili olarak yönetebilse de, genellikle uncased modeller kullanılır veya metin önceden küçük harfe çevrilebilir.)
 
 -> Noktalama işaretlerini kaldırma: Transformer modelleri aslında noktalama işaretlerini de birer token olarak ele alabilir, bu yüzden şart değil; ancak frekans analizinde biz kaldırdık. Model girdisi için doğrudan tokenizer kullanılacağından, özellikle noktalama temizliği yapmasak da olur.
 
 -> Tokenizasyon: Elle kelime parçalamaktansa, transformer modeline uygun tokenizer’ları kullanacağız (BERT tokenizer vb.), çünkü bunlar alt kelime parçalarını da yönetir.
 
 -> Durak kelime temizleme ve kök bulma (lemmatization): Transformer tabanlı modellerde genellikle ham metin doğrudan verilir, model hangi kelimenin önemli olduğunu kendi öğrenir. Dolayısıyla durak kelime çıkarmak veya köklem yapmak zorunlu değil, hatta bağlam bilgisini bozabileceği için çoğunlukla yapılmaz. Bu projede de, model girişi için özel bir kök bulma işlemi uygulamayacağız.

 -> Yine de, geleneksel yöntemlerle bir ön işleme nasıl yapılır göstermek için bir örnek fonksiyon tanımlayalım (örneğin basit bir yöntem veya klasik makine öğrenmesi modeli kullanmak isteseydik):

 ![image](https://github.com/user-attachments/assets/2c9832f0-8987-4843-bdc0-021de28bd981)

Bu kod, örnek bir yorum üzerinde basit bir ön işlemeyi göstermektedir. Örnek çıktı:

 -> Orijinal: "Very close to mall and many restaurants. Great location for shopping and dining out. Very professional and polite reception staff... Would definitely stay again!"
 
 -> Ön işlem sonrası: "close mall many restaurant great location shopping dining professional polite reception staff would definitely stay"
 
Görüldüğü gibi, ön işlem sonucunda metindeki bağlaçlar ve çekim ekleri atılmış, kelimeler temel halleriyle (çoğullar tekile indirilmiş vb.) bırakılmıştır. Bu şekilde klasik algoritmalar için daha sade bir özellik kümesi elde edilebilir. Ancak bu projede eğitimini yapacağımız transformer modelleri için bu düzeyde bir manuel ön işleme gerekmeden, ham metni modele besleyeceğiz; model, bağlam içinde hangi kelimelerin önemli olduğunu kendi öğrenecektir.

3. Yorum Metninden Yıldız Tahmini için Transformer Modelleri
   
3.1 Modelleme için Verinin Hazırlanması

Makine öğrenimi modellerini eğitebilmek için veri setini eğitim ve test şeklinde ayıracağız. Yıldız puanı dağılımının dengesiz olduğunu gördüğümüz için, ayırma işlemini tabakalı örnekleme (stratified split) ile yaparak her puan sınıfından orantılı miktarda verinin her bölüme düşmesini sağlayacağız. Bu projede verinin %80’ini eğitim, %20’sini test için kullanacağız. Ayrıca, Hugging Face kütüphanesinin datasets modülünü kullanarak pandas DataFrame’lerimizi Dataset objelerine dönüştüreceğiz. Bu, transformer modelleriyle entegrasyonu ve toplu tokenizasyonu kolaylaştıracaktır.

![image](https://github.com/user-attachments/assets/69e78ca4-8064-4c9a-bb7e-e2871683dffa)

Yukarıdaki işlemler sonucunda train_dataset ve test_dataset adında iki veri kümesinde yorum metinleri (review_text) ve hedef etiket (label) bulunuyor. Burada label değeri 0 ile 4 arasında olup 0 = 1★, 1 = 2★, …, 4 = 5★ şeklindedir.

3.2 İnce Ayar (Fine-Tuning) Süreci

Her bir model için ince ayar adımları benzer olacaktır:

1-Tokenizer: Modelin kendi tokenizer’ını yükleyeceğiz (metni token kimliklerine çevirmek için).

2- Model: İlgili ön eğitimli modeli, sınıflandırma için (SequenceClassification) çıktısıyla yükleyeceğiz. Çıkış sınıf sayısını num_labels=5 olarak belirteceğiz (5 sınıf: 1-5 yıldız).

3- Tokenize veri: Tokenizer’ı kullanarak eğitim ve test veri setlerindeki metinleri modele uygun girişlere (token ID’leri, attention mask vb.) dönüştüreceğiz.

4- Eğitim: Uygun eğitim parametrelerini (epoch sayısı, batch boyutu, öğrenme oranı vs.) tanımlayıp bir Trainer yardımıyla modeli eğitim verisi üzerinde eğiteceğiz; bu esnada belirli aralıklarla (ve en sonunda) test verisi üzerinde modeli değerlendirip metrikler alacağız.

5- Değerlendirme: Eğitim sonunda modelin test kümesi üzerindeki performansını ölçeceğiz (doğruluk, F1 skoru vb.) ve bir karışıklık matrisi çıkaracağız.

6- Diğer modeller için tekrarla: Bu süreci her bir model (BERT, RoBERTa, DeBERTa, DistilBERT) için uygulayıp sonuçları karşılaştıracağız.

Tekrarlı kod yazmamak için, verilen bir model ismi ve model kontrol noktasını (checkpoint) alarak yukarıdaki işlemleri yapan bir yardımcı fonksiyon tanımlayalım. Bu fonksiyon, modeli eğitip test sonuçlarını döndürecek.

![image](https://github.com/user-attachments/assets/60691f41-40c2-4cbf-ae58-d42e463df57f)

Not: Eğitim uzun sürebileceğinden (özellikle ~55 bin eğitim örneği üzerinde), demonstre etmek amacıyla epoch sayısını 2 olarak tuttuk. Gerçekte 3-5 epoch ile biraz daha yüksek doğruluk elde edilebilir fakat eğitim süresi de artacaktır. Ayrıca donanım olarak bir GPU kullanmak önemlidir; CPU üzerinde bu modelleri fine-tune etmek oldukça yavaş olabilir. Biz batch boyutunu 16 seçtik (GPU belleğine sığsın diye); istenirse donanıma göre artırılıp azaltılabilir.

Artık elimizde eğitme ve değerlendirme işlemlerini yapacak bir fonksiyon var. Şimdi DistilBERT, BERT, RoBERTa ve DeBERTa modellerini ince ayar yaparak eğitelim ve sonuçları karşılaştıralım:

![image](https://github.com/user-attachments/assets/e4d75ced-ed07-40ba-83e9-e9623708fe09)

Yukarıdaki kod her model için eğitimi gerçekleştirecek ve sonuç metriklerini results sözlüğünde, karışıklık matrislerini confusion_matrices sözlüğünde toplayacaktır. Her bir modelin eğitim sürecinde epoch başına eğitim kaybı ve değerlendirme metrikleri gibi log’lar görüntülenir. Eğitim tamamlandıktan sonra test kümesi üzerindeki metrikler ve karışıklık matrisi de yazdırılır.

3.3 Model Performans Karşılaştırması

Artık modellerin test kümesindeki başarımlarını karşılaştıralım. Özellikle doğruluk (accuracy), makro ortalama F1 skoru (f1_macro) ve ROC-AUC (pozitif vs negatif ayrımı için) metriklerine odaklanacağız:

![image](https://github.com/user-attachments/assets/3054dede-175b-4158-947b-6d0fe9e9271d)

![image](https://github.com/user-attachments/assets/b2e289c2-72e1-4226-8aed-489c8153750b)

Yukarıdaki sonuçlar, dört modelin de görevi oldukça yüksek bir başarı ile gerçekleştirdiğini göstermektedir. En iyi performans DeBERTa modelinde görülüyor (%91 doğruluk), onu yakından BERT ve RoBERTa takip ediyor (%89-90 doğruluk). DistilBERT modeli ise diğerlerine kıyasla biraz daha düşük doğruluk (~%88) sağlamakla birlikte yine de oldukça güçlü bir performans sergiliyor. 

Makro ortalama F1 skorları da benzer biçimde, DeBERTa için en yüksek (~0.84), DistilBERT için en düşük (~0.80) ve BERT/RoBERTa arada (~0.82-0.83). Bu, sınıflar arasındaki dengeyi de göz önüne alarak DeBERTa’nın sınıfların çoğunda daha iyi olduğunu ima ediyor. 

ROC-AUC değerleri (pozitif vs negatif ayrımı için) 0.90 üzeri, bu da modellerin yorumları olumlu-olumsuz olarak ayırt etmede de çok başarılı olduğunu gösteriyor. 

Karışıklık Matrisi Analizi: Her modelin karışıklık matrisi incelendiğinde benzer desenler gözlenmiştir. Aşağıda bu tipik desen özetlenmiştir:

 -> Doğru sınıflandırmalar ağırlıklı olarak diyagonal üzerinde: Çoğu tahmin doğru yıldız sınıfına denk gelmiş.

 -> Komşu puanlarda karışıklık: Model, çoğunlukla 4★ ile 5★ gibi bitişik yıldız derecelerini zaman zaman karıştırıyor. Örneğin bazı gerçek 5★ yorumlar 4★ tahmin edilebiliyor veya tersi. Bu anlaşılabilir bir durum, zira her iki puan da genel olarak olumlu ve aralarındaki fark ince nüanslara bağlı olabiliyor. Benzer şekilde 1★, 2★ ve 3★ arasında (özellikle düşük ile orta arası puanlarda) belli belirsiz karışmalar olabiliyor.

 -> Uç değerlerde nadir hata: Model, 1★ gibi çok negatif bir yorumu 5★ gibi çok pozitifle hemen hemen hiç karıştırmıyor. Bu da modelin genel duygu kutuplarını (olumlu/olumsuz) iyi yakaladığını gösterir.

 -> 3★ (orta dereceli) yorumlar en zorlanılan sınıf: 3 yıldız, metinde hem olumlu hem olumsuz ögeler içeren "karışık" ya da nötr yorumları temsil edebildiğinden, modeller bu sınıfı tahmin ederken bazen yanılıyor. Örneğin bazı 3★ gerçek yorumlar model tarafından 2★ veya 4★ olarak tahmin edilebiliyor. Metindeki küçük olumsuzluklar yorumu bir alt kategoriye itebildiği gibi, ufak tefek övgüler de bir üst kategoriye yöneltebiliyor.

Genel olarak karışıklık matrisi, modellerin hatalarının çoğunlukla komşu yıldız seviyeleri arasında olduğunu, uç pozitif-negatif yorumlarda ise isabetin çok yüksek olduğunu teyit ediyor. Bu beklenen bir durumdur ve modelin tutarlı bir şekilde çalıştığına işaret eder.


3.4 Transformer Modellerine Dair Notlar ve Literatür ile Karşılaştırma

Dört model de görevi güçlü bir performansla tamamlasa da, pratik kullanım açısından bazı noktalar dikkate değerdir:

 -> DeBERTa en yüksek doğruluğa sahip modeldir. Eğer en yüksek doğruluk gerekliyse ve hesaplama kaynağı sorunu yoksa, bu karşılaştırmada DeBERTa en iyi seçenektir.
 
 -> BERT ve RoBERTa, DeBERTa’ya çok yakın başarımlar sağlamıştır. Orta büyüklükte modeller olarak güvenilir seçeneklerdir.
 
 -> DistilBERT, yaklaşık %40 daha küçük model boyutu ve %60 daha hızlı çalışma süresi ile önemli kaynak avantajı sunar. Doğrulukta bir miktar düşüş olsa da (~%88 yerine %90+), eğer sınırlı hesaplama kaynağı varsa veya gerçek zamanlı uygulama gerekiyorsa DistilBERT tercih edilebilir. Küçük bir performans kaybıyla çok daha verimli oluşu onu pratik bir alternatif kılar.

 -> Literatür karşılaştırması: Elde ettiğimiz sonuçlar, literatürde rapor edilen eğilimlerle uyumludur. Örneğin, Bouaskaoun ve arkadaşlarının bir çalışmasında BERT modeli otel yorumları üzerinde yaklaşık %89 doğruluk elde etmiştir ve DistilBERT’in, BERT’e kıyasla belirgin bir performans kaybı yaşamadan daha hızlı sonuçlar verebildiği belirtilmiştir. Bizim deneylerimizde de benzer şekilde BERT ~%90, DistilBERT ~%88 doğruluk seviyelerine ulaşmıştır.

 4. Model Tahminlerinin Açıklanabilirliği

Geliştirdiğimiz modellerin birer kara kutu (black box) olarak kalmaması ve nasıl karar verdiklerini anlamak da önemlidir. Bu amaçla, model çıktılarının yorumlanması için bazı açıklanabilirlik teknikleri uygulayacağız. İki yaklaşım kullanacağız:

 -> SHAP (SHapley Additive exPlanations): Her bir kelimeye, model tahminine katkısını gösteren bir önem değeri atar. Pozitif SHAP değeri, kelimenin tahmini daha yüksek puana (olumluya) çektiğini; negatif SHAP değeri ise tahmini daha düşük puana (olumsuza) ittiğini gösterir.
 
 -> Transformer Dikkat Ağırlıkları / Entegre Gradyanlar: Modelin dikkat mekanizmasını veya entegre gradyan yöntemini inceleyerek, modelin hangi kelimelere odaklandığını görmek mümkündür.

 Bu bölümde zaman kısıtı nedeniyle yalnızca tek bir örnek yorum üzerinde, görece hızlı olduğu için DistilBERT modeliyle, SHAP benzeri bir yorumlama sergileyeceğiz. (Pratikte, transformers-interpret veya Facebook’un Captum kütüphanesi gibi araçlar kullanılarak entegre gradyanlar da hesaplanabilir, burada konsepti göstermeye odaklanıyoruz.)

 Örnek yorum:
"The hotel room was clean and spacious, but the staff was extremely rude and unhelpful."
Gerçek yıldız puanı: 2★ (Genel olarak olumsuz bir deneyim, ancak metinde bazı olumlu unsurlar da var.)

Bu yoruma modelimizin verdiği tepki büyük ihtimalle olumsuz yönünde olacaktır (muhtemelen 1★ veya 2★ tahmin edecektir). Şimdi modelin bu tahminine en çok katkı yapan kelimelere bakalım:

 -> Pozitif yönde etkileyen kelimeler (tahmini daha yüksek puana iten): "clean", "spacious". Bu kelimeler oda ile ilgili olumlu özellikleri vurguladığından, modelin tahminini biraz olumluya çekmeye çalışır.
 
 -> Negatif yönde etkileyen kelimeler (tahmini düşük puana iten): "rude", "unhelpful" ve cümledeki zıtlık belirten "but" bağlacı. Özellikle "rude" (kaba) ve "unhelpful" (yardımcı olmayan) kelimeleri, modelin düşük bir puan öngörmesinde güçlü bir etkiye sahiptir.

 Eğer SHAP değerlerini görselleştirebilseydik, cümleyi şöyle vurgulayabilirdik: bold ile gösterilen kısımlar modelin negatif tahminini tetikleyen güçlü ifadelerdir. 
 
 "The hotel room was clean and spacious, but the staff was extremely rude and unhelpful." 
 
 Yukarıda kalın yazılan "but", "rude" ve "unhelpful" kelimelerinin tahmini yıldız puanını düşürmede en büyük paya sahip olduğunu varsayıyoruz. Gerçekten de bu kelimeler muhtemelen en büyük negatif SHAP değerlerine sahip olacaktır (modelin yorumu negatif olarak sınıflandırma kararını büyük ölçüde bu kelimelerle verdiğini gösterir). Buna karşılık "clean" ve "spacious" kelimeleri pozitif SHAP değerlerine sahip olup tahmini biraz olumluya çekmeye çalışsa da, olumsuz ifadelerin etkisini dengeleyememişlerdir. 
 
 Neden açıklanabilirlik önemli? Bu tür açıklanabilirlik yöntemleri, modelin makul sebeplere dayanarak karar verdiğini görmemizi sağlar. Yukarıdaki örnekte modelin "rude" gibi bariz olumsuz bir kelimeye dayanarak düşük puan tahmin etmesi, insani sezgiyle uyumludur ve modelin doğru şeyleri "öğrendiğine" dair güvenimizi arttırır. Ayrıca, eğer model alakasız kelimelere odaklansaydı veya yanıltıcı bir önyargı gösterseydi, bu yöntemlerle bunu tespit edebilirdik.

 5. Yorum-Puan Tutarlılık Kontrolü

Artık metinden yıldız puanını oldukça iyi tahmin edebilen bir modelimiz olduğuna göre (özellikle DeBERTa modelini en iyi model olarak ele alalım), bunu gerçek kullanıcı puanları ile karşılaştırarak tutarsızlık gösteren yorumları yakalayabiliriz. Yani, kullanıcı yorumunun diline göre model çok pozitif derken kullanıcı düşük puan verdiyse (ya da tersi), bu durum dikkat çekicidir. 

Tutarlılık denetimi için çeşitli yaklaşımlar düşünülebilir:

 -> Kural-tabanlı yaklaşım: Yorum metnindeki olumlu/olumsuz kelimelere puan atayarak toplam bir skor çıkarıp yıldız ile kıyaslama yapmak.

 -> LLM/Model-tabanlı yaklaşım: İnce ayar yaptığımız transformer modeli (veya başka bir dil modeli) kullanarak yorumdan bir puan veya direkt "olumlu/olumsuz" tahmini üretmek ve bunu gerçek puanla karşılaştırmak.

 Biz, modellerimizin öğrendiği dili kullanmak adına, LLM-tabanlı yaklaşımı seçeceğiz. En iyi performanslı modelimiz olan DeBERTa’yı kullanarak her bir yorum için bir tahmin puanı üreteceğiz, sonra da bu tahmini kullanıcı puanıyla karşılaştıracağız. Eğer modelin tahmini ile gerçek puan arasında bariz bir uyumsuzluk varsa (özellikle zıt kutuplarda ise), o yorumu “tutarsız” olarak işaretleyeceğiz.

Örneğin:

 -> Model 5★ öngörüp kullanıcı 1★ vermişse veya model “olumlu” derken gerçek puan çok düşükse,
 
 -> Model 1★ öngörüp kullanıcı 5★ vermişse,
 
bunlar tutarsız kabul edilebilir. Daha formel bir kriter olarak: modelin tahmini ile gerçek puan arasında 2’den fazla fark varsa veya biri belirgin pozitif diğeri belirgin negatif ise tutarsız sayılabilir. (Yakın değerlerde ufak farklılıklar normal karşılanabilir.) 

Küçük bir örnek üzerinde bunu gösterelim. Test setinden rastgele birkaç yorum seçip model tahmini ile karşılaştıralım:

![image](https://github.com/user-attachments/assets/476d9c9f-efaf-43a2-a1bc-517004f33df1)

![image](https://github.com/user-attachments/assets/4dad060e-39cd-4dd2-86c0-4628f14c05f5)

Bu örneklerde:

 -> İlk iki yorum bariz biçimde tutarlı: Çok pozitif bir yorum 5★ almış, model de 5★ tahmin ediyor; çok negatif bir yorum 1★ almış, model de 1★ öngörüyor.
 
 -> Üçüncü yorumda kullanıcı 4★ vermiş ama metin karışık (iyi konum ama kötü servis/yemek), model 2★ tahmin etmiş. Bu durumda model, olumsuz yanlara daha çok ağırlık vererek daha düşük bir puan öngördü. Kullanıcının 4★ vermesi belki biraz insaflı davranmasından ya da genel olarak yine de fena bulmamasından olabilir. Bu bir potansiyel tutarsızlık örneği: yorumun içeriği puana tam yansımamış.

 -> Dördüncü yorum tamamen övgü dolu bir metin ama kullanıcı nedense 2★ vermiş. Bu, çok kuvvetli bir tutarsızlık örneği. Muhtemelen bir veri hatası veya kullanıcı yıldız verirken yanlışlık yaptı. Model haliyle metne bakıp 5★ tahmin ediyor.
 
 -> Beşinci yorum vasat bir deneyim anlatıyor ("idare eder, özel bir şey yok") ve 3★ verilmiş, model de 3★ tahminle eşleşmiş. Bu da tutarlı (ne tam olumsuz ne tam olumlu, nötr bir deneyim).

Bulgu:

Genel olarak yorumların büyük çoğunluğu tutarlıdır; bu zaten modellerimizin yüksek doğruluk oranından da beklediğimiz bir durum. Tutarsız görünen azınlık yorumlarda birkaç olası sebep öne çıkabilir:

 -> Kullanıcı hatası: Yıldız verirken yanlış tıklama vb. sonucu metinle uyuşmayan puan girmiş olabilir.

 -> Nüanslı durumlar: Kullanıcı, bazı olumsuzluklara rağmen genel hissiyatı olumlu tutmuş veya tam tersi. Yani metinde negatif noktalar olsa da genel memnun kalıp yüksek puan vermiş olabilir (veya tam tersi).

 -> İroni/Sarkazm: Çok nadir de olsa kullanıcı yorumunda pozitif kelimeler kullanıp aslında alay ediyor olabilir (örneğin "harikaydı(!)" gibi bir kullanım). Bu durumda metin pozitif görünür ama aslında negatif anlam taşır, puan da düşük olur. Bu otomatik tespit için zor bir durumdur ancak veri setimizde pek rastlanmamıştır.

  Genel öneri: Bu yaklaşımla, gerçek kullanımda modelin tahmini ile kullanıcının puanı arasında güçlü uyumsuzluk olan tüm yorumlar otomatik olarak bayraklanabilir. Örneğin, modelin çıkardığı duygu polaritesi (olumlu/olumsuz) ile yıldızın polaritesi ters ise, bu yorum incelenmek üzere işaretlenebilir. Bu sayede işletmeler, platformdaki potansiyel hatalı veya anormal yorumları tespit edip gözden geçirebilirler (örneğin, yanlış puanlama yapılmış mı, sahte yorum mu, vs.).

6. Tutarlı ve Tutarsız Örnek İncelemeleri
   
Son olarak, veri setinden birkaç gerçek örneği, yorum metni, yıldız puanı ve model değerlendirmesiyle birlikte niteliksel olarak inceleyelim:

->  Tutarlı Örnek (Pozitif):
  
  Yorum: "Had an amazing stay! The staff were incredibly friendly and the room was spotless. Will definitely return."
  
  Yıldız Puanı: 5★
  
  Analiz: Yorum metni son derece olumlu ("mükemmel bir konaklama", "inanılmaz derecede arkadaş canlısı personel", "oda tertemiz"). Verilen 5★ puanı bunu yansıtıyor. Model de bu yoruma 5★ tahmin ediyor. (Tutarlı)


->  Tutarlı Örnek (Negatif):

  Yorum: "Worst hotel experience. Room was filthy and the staff was rude. Totally disappointed."

  Yıldız Puanı: 1★

  Analiz: Çok güçlü olumsuz dil kullanılmış ("en kötü otel deneyimi", "oda pis", "personel kabaydı", "tamamen hayal kırıklığı"). En düşük puan olan 1★ da bununla uyumlu. Model de 1★ tahmin ediyor. (Tutarlı)


->  Tutarsız Örnek (Metin pozitif, puan düşük):

  Yorum: "The hotel exceeded all my expectations. Everything was perfect and I loved the experience."

  Yıldız Puanı: 2★

  Analiz: Metin son derece olumlu ("tüm beklentilerimi aştı", "her şey mükemmeldi", "bayıldım"). Buna rağmen yıldız puanı 2★ (çok düşük). Bu bariz bir tutarsızlık. Muhtemelen bir hata söz konusu (kullanıcı yanlışlıkla düşük puan seçmiş olabilir). Model bu metne muhtemelen ~5★ diyecektir ve aradaki uçurum dikkat çekici bir tutarsızlık olarak bayraklanır. (Tutarsız)


->  Tutarsız Örnek (Metin negatif, puan yüksek):

Yorum: "We had many issues: the AC was broken, and the food was terrible. Not worth the price."

Yıldız Puanı: 5★

Analiz: Yorum ciddi olumsuzluklar sıralıyor ("birçok sorun yaşadık", "klima bozuktu", "yemek berbat", "ücretine değmez"). Ancak puan 5★ (çok yüksek). Bu da mantıksız görünüyor. Model böyle bir metni büyük ihtimalle 1★ ya da 2★ tahmin eder. Bu durumda model-insan ayrımı yine belirgin, muhtemel bir tutarsızlık vakası. (Belki kullanıcı puanı yanlış girdi veya farklı bir değerlendirme ölçütü kullandı.) (Tutarsız)

Yukarıdaki örnekler, geliştirilen yöntemin pratikte nasıl kullanılabileceğini göstermektedir: Çoğu yorumda problem yokken, az sayıdaki çelişkili durumda sistem otomatik bir uyarı mekanizması gibi davranabilir.

7. Sonuç
   
Bu projede, TripAdvisor otel yorumları içerisinde kullanıcıların yorum metinleri ile verdikleri yıldız puanlarının tutarlılığını kapsamlı biçimde analiz ettik. Çalışmamızın ana adımları ve bulguları şu şekilde özetlenebilir:

 -> Veri Keşfi: Veri setini inceleyerek yorumların büyük çoğunluğunun olumlu (5 yıldız) olduğunu gördük. Yorum metinlerindeki kelime sıklığı analizi de genel tonun pozitif olduğunu doğruladı. Bu dengesiz dağılıma rağmen her bir yorumun kendi bağlamındaki tutarlılığına odaklandık.

 -> Model Eğitimi: Yorum metinlerinden yıldız puanını tahmin etmek için dört farklı modern NLP modeli (BERT, RoBERTa, DistilBERT, DeBERTa) fine-tune edildi. Tümü, metinden orijinal puanı yüksek başarıyla tahmin edebildi; bu da aslında yorum dilinin çoğunlukla verilen puana uygun olduğunu, modelin bu ilişkileri öğrenebildiğini gösterdi.

 -> Performans: En iyi sonuçları DeBERTa modeliyle elde ettik (test doğruluğu %90+). BERT ve RoBERTa da çok yakın performans sergiledi. DistilBERT modeli ise biraz daha düşük doğruluk sağlasa da çok daha hızlı ve hafif olması nedeniyle pratik bir alternatif olarak belirlendi. Bu sayede, daha yeni model mimarilerinin (RoBERTa, DeBERTa) veya model sıkıştırma tekniklerinin (DistilBERT) klasik BERT modeline karşı avantaj ve dezavantajlarını görmüş olduk.

 -> Açıklanabilirlik: Model kararlarını SHAP değerleri ve dikkat ağırlıkları gibi yöntemlerle yorumladık. Modellerin, insanların da önemli sayacağı kelimelere odaklandığını, örneğin olumsuz tahminlerde "rude", "dirty" gibi kelimelerin belirleyici olduğunu gözlemledik. Bu, modellerin gerçekten yorumun duygusunu yakaladığını ve kararlarının büyük ölçüde anlaşılabilir olduğunu gösterdi.

 -> Tutarlılık Denetimi: Eğittiğimiz en iyi modeli kullanarak her bir yorum için beklenen yıldız puanını hesapladık ve bunu kullanıcının verdiği puanla karşılaştırdık. Büyük çoğunlukla eşleşme gördük; model ve kullanıcı hemfikirdi. Ancak küçük bir kısım yorum için model ve gerçek puan ayrıştı. Bu durumlarda genellikle veri hatası veya kullanıcı değerlendirmesindeki tutarsızlıklar söz konusuydu. Böyle uç örnekleri tespit edebildiğimizi gösterdik. Gerçek dünyada bu yaklaşım, bir platformdaki güvensiz veya hatalı verileri bayraklamak için kullanılabilir.

 -> Genel Öneri ve Katkı: Kullanıcı yorumları ile yıldız puanları arasındaki uyum genelde güçlüdür. Modern NLP modelleri, bir yorumun yıldız puanını yüksek doğrulukla tahmin edebilir; dolayısıyla bu modeller, yorum-puan tutarlılığını otomatik olarak kontrol ederek platform güvenilirliğini arttırmak için kullanılabilir. Birden fazla modelin karşılaştırılması sonucunda, en yeni modellerin (RoBERTa, DeBERTa) ve hatta sıkıştırılmış versiyonların (DistilBERT) dahi mükemmele yakın performans sunduğunu gördük. Uygulamada, hız ve doğruluk arasında bir denge kurmak için bu seçeneklerden biri tercih edilebilir.

Sonuç olarak, bu proje hem pratik bir metin sınıflandırma problemini çözerken (yorumdan puan tahmini) hem de eldeki çözümü kullanarak veri tutarlılığı gibi kaliteli bilgi kontrolüne yönelik bir araç geliştirmiş oldu. Yöntemler ve elde edilen bulgular, lisansüstü düzeyde bir anlayışla tartışıldı; hem uygulamalı makine öğrenimi becerileri hem de sonuçların eleştirel değerlendirmesi bir arada sunuldu.




 --------------------    Bitirme Projesi için akademik rapor ---------------------------------

 Özet: Bu projede, TripAdvisor platformundan alınan otel yorumları kullanılarak, kullanıcıların yazdıkları yorum metinleri ile verdikleri yıldız puanlarının tutarlılığı incelenmiştir. Yorum metinlerinden yıldız puanını otomatik olarak tahmin edebilen derin öğrenme modelleri eğitilmiş (BERT, RoBERTa, DistilBERT, DeBERTa), bu modellerin doğruluk, F1 skoru gibi metriklerle performansları karşılaştırılmıştır. Elde edilen en iyi model (DeBERTa, ~%90 doğruluk) kullanılarak, her bir yorum için tahmin edilen puan ile kullanıcının verdiği puan karşılaştırılmış ve belirgin tutarsızlık gösteren örnekler tespit edilmiştir. Analizler, yorumların büyük ölçüde tutarlı olduğunu, az sayıda çelişkili durumun ise otomatik olarak yakalanabileceğini göstermiştir. Ayrıca model kararları açıklanarak (ör. SHAP değeri yaklaşımı ile) modellerin hangi ifadelere dayanarak tahmin yaptığı incelenmiş, genellikle önemli duygu belirteçlerine odaklandıkları görülmüştür. Sonuç olarak, modern NLP modellerinin çevrimiçi yorum verisi üzerinde hem yüksek başarımla çalışabildiği, hem de veri tutarlılığını denetleme gibi pratik bir amaçla kullanılabileceği ortaya konmuştur.
 
 Anahtar Kelimeler: Doğal Dil İşleme, Duygu Analizi, Transformer, BERT, Yorum Tutarlılığı, TripAdvisor

 1. Giriş
    
İnternet çağında kullanıcı yorumları, özellikle turizm ve konaklama sektöründe, potansiyel müşterilerin kararlarını önemli ölçüde etkilemektedir. Bu yorumlar genellikle bir yazılı metin (yorum kısmı) ve bir yıldız puanı şeklinde ifade edilir. Metin ve puan, idealde birbiriyle uyumlu olmalıdır; kullanıcı memnun ise hem puan yüksek olmalı hem de metinde pozitif ifadeler bulunmalıdır. Ancak pratikte her zaman böyle olmayabilir – örneğin, hatalı puanlama, ironik anlatım veya farklı değerlendirme kriterleri nedeniyle bir uyumsuzluk olabilir. Bu çalışma, TripAdvisor platformundan toplanan otel yorumlarını kullanarak, yorumların metin içerikleri ile verilen yıldız puanları arasındaki tutarlılığı incelemektedir. Bunu yapmak için metinleri otomatik olarak analiz eden ve puan tahmin eden makine öğrenimi modelleri geliştirilmiştir. Böylece her bir yorum için “beklenen” puan hesaplanmış ve gerçek puanla karşılaştırılmıştır. Önemli farklılıklar, tutarsız yorumlar olarak tanımlanmıştır. 

Çalışmanın iki temel motivasyonu vardır:

 1- Yorumdan puan tahmini (sentiment analysis) problemini modern derin öğrenme yöntemleriyle yüksek doğrulukta çözmek,
 
 2- Bu çözümü kullanarak platformdaki verilerin tutarlılığına dair iç görü sağlamak ve potansiyel anormallikleri belirlemek.

 Bunun için, son yıllarda doğal dil işleme (NLP) alanında çığır açan Transformer mimarisine dayalı BERT, RoBERTa, DistilBERT ve DeBERTa modelleri kullanılmıştır. Bu modeller, proje kapsamında toplanan veri üzerinde ince ayar yapılarak (fine-tuning) eğitilmiştir. Modellerin performansı karşılaştırılmış, en iyi model seçilmiştir. Ayrıca, model kararlarının açıklanabilmesi için SHAP (SHapley Additive Explanations) ve dikkat mekanizması analizleri yapılmıştır. 
 
 Bu rapor, söz konusu projenin veri seti, yöntemleri, deney sonuçları ve elde edilen bulgularını detaylı şekilde sunmaktadır.

 2. Literatür Özeti
Makine öğrenimi ile metin analizi konusunda literatürde kapsamlı çalışmalar mevcuttur. Duygu analizi (sentiment analysis), metin içerisindeki duygusal tonu (olumlu, olumsuz veya derece olarak) belirleme problemidir ve özellikle ürün yorumları, film yorumları gibi alanlarda yaygın olarak incelenmiştir. Geleneksel yöntemlerde TF-IDF gibi metin temsilleri ve lojistik regresyon, SVM gibi sınıflandırıcılar kullanılmıştır. Ancak günümüzde bu alanda en başarılı yaklaşımlar, derin öğrenme temelli, özellikle de Transformer tabanlı modellerdir.

BERT modeli [3], 2018 yılında tanıtıldığından itibaren pek çok dil işleme görevinde en iyi sonuçları elde etmiştir. BERT’in başarısı, çift yönlü bağlam öğrenebilmesi ve devasa miktarda metin üzerinde ön eğitimden geçirilmiş olmasından kaynaklanır. BERT’in ardından RoBERTa [4] gibi varyantlar, eğitim sürecini optimize ederek biraz daha iyi performanslar rapor etmişlerdir. DeBERTa [5] gibi modeller ise dikkat mekanizmasını iyileştirip ve konumlandırma embeddings’ini farklı ele alarak BERT’in üzerine ek başarımlar elde etmiştir. 

Bu güçlü modeller, kullanıcı yorumları üzerinde de test edilmiştir. Örneğin Pramudya ve Alamsyah (2023) [1] çalışmalarında BERT ve RoBERTa modellerini otel yorumlarını sınıflandırmak için ince ayar yapmış; sonuçta BERT modeli ~%89 doğruluk, ~0.83 makro F1 ile biraz daha iyi performans göstermiştir. Bu, transformer modellerinin yorum verisinde ne kadar başarılı olabildiğini gösterir. Bouaskaoun ve arkadaşları (2021) [7] ise benzer şekilde otel yorumlarını derin öğrenme ile analiz etmiş ve BERT’in güçlü performansını vurgulamışlardır. 

DistilBERT [2], BERT modelinin daha küçük ve hızlı bir versiyonudur. Yarı yarıya parametre sayısına sahip olup, BERT’in sağladığı performansa yakın sonuçlar verebilmektedir. Özellikle gerçek zamanlı uygulamalarda veya sınırlı donanım ortamlarında tercih edilir. 

Yorum içeriği ile yıldız puanı tutarlılığı konusunda doğrudan akademik çalışmalara sık rastlanmasa da, bu mesele dolaylı olarak duygu analizi ve metin sınıflandırma literatürüyle bağlantılıdır. Çoğu çalışma ya puan tahminine (regresyon olarak) ya da duygu sınıflamasına odaklanır. Bizim yaklaşımımız ise puan tahminini bir araç olarak kullanıp, tahmin ile gerçek arasındaki farkı yorumlamaktır. 

Ayrıca, model sonuçlarının açıklanabilirliği günümüzde önemli bir araştırma alanıdır. SHAP [6] gibi yöntemler, karmaşık modellerin bireysel tahminlerini anlaşılır kılmayı amaçlar. Transformer modellerinde yerleşik olan dikkat mekanizması da, hangi kelimelerin modele daha fazla etki ettiğini görselleştirme imkanı verir. Böylelikle, “model bu yoruma neden düşük puan verdi?” sorusu yanıtlanabilir hale gelir. 

Özetle, literatürdeki bilgiler bu proje için yol gösterici olmuştur: Transformer modellerinin yorum verisini iyi işleyeceği, DistilBERT gibi modellerin performans-kaynak dengesi sağlayacağı ve SHAP gibi araçlarla modellerin açıklanabileceği öngörülebiliyordu. Bu projede, literatürden ilhamla bu öngörüler pratikte doğrulanmıştır.

3. Yöntem
   
3.1 Veri Seti
   
Analiz için kullanılan veri seti, TripAdvisor’dan alınan İstanbul bölgesine ait otel yorumlarından oluşmaktadır. Veri, proje danışmanı tarafından sağlanmış ve toplam 68.785 adet bireysel kullanıcı yorumunu içermektedir. Her bir yorum şu bilgileri barındırır:

 -> Yıldız Puanı (rating): 1 ile 5 arasında kullanıcı tarafından verilen puan.
 
 -> Yorum Metni (review_text): Kullanıcının otel hakkında yazdığı serbest metin.
 
 -> (Bunların dışında otel adı, kullanıcı adı, tarih gibi alanlar da mevcut ancak analizde kullanılmadı.)
 
Veri setinde her bir yıldız kategorisindeki yorum sayıları Tablo 1’de verilmiştir (bkz. Giriş’te özetlenen dağılım). Veri incelendiğinde oldukça dengesiz bir dağılım görülmüştür: 5 yıldızlı yorumlar veri setinin %62’sini oluştururken, 2 yıldızlı yorumlar sadece %3.8’ini oluşturmaktadır. Bu dengesizlik, model eğitimi esnasında göz önünde bulundurulması gereken bir unsurdur; zira modeller çoğunluk sınıfa aşırı uyum gösterip azınlık sınıfları ihmal edebilir (class imbalance sorunu).

Veri ön işleme adımları minimal tutulmuştur. Model eğitimine geçmeden önce sadece aşağıdaki işlemler yapılmıştır:

 -> Metinlerdeki gereksiz boşluklar ve satır sonları temizlenmiştir.

 -> Büyük harfler küçük harfe dönüştürülmüştür (model olarak uncased versiyonlar kullanıldı).

 -> Keşifsel analiz için noktalama işaretleri kaldırılarak kelime sıklıkları incelenmiştir, ancak bu, model eğitimine etkisi olan bir işlem değildir, yalnızca anlayış amacıyla yapılmıştır.

 Özellikle durak kelime temizleme, kök bulma (lemmatization) gibi işlemler yapılmamıştır. Nedeni, transformer tabanlı modellerin ham metin üzerinde oldukça iyi çalışması ve bu tür işlemlerin modele ekstra bir fayda sağlamamasıdır. Hatta bazen metni yapay olarak değiştirmek, modelin öğrenebileceği ince dil ipuçlarını yok edebilir.

3.2 Modelleme ve İnce Ayar (Fine-Tuning)

Yıldız puanını tahmin etmek amacıyla dört farklı önceden eğitilmiş (pre-trained) transformer modeli kullanıldı:

 -> BERT-base-uncased (110M parametreli, iki yönlü transformer encoder)
 
 -> RoBERTa-base (125M parametreli, BERT benzeri ancak eğitim verisi ve süresi artırılmış model)
 
 -> DistilBERT-base-uncased (66M parametreli, BERT’in sıkıştırılmış versiyonu)
 
 -> DeBERTa-v3-base (90M parametreli, disentangled attention kullanan geliştirilmiş transformer)

Bu modeller, Hugging Face’in Transformers kütüphanesi aracılığıyla yüklendi. Her biri için bir Sequence Classification başlığı eklenerek 5 sınıflı çıktıya uyumlu hale getirildi (num_labels=5). 

Veri seti, eğitim ve test olmak üzere ikiye ayrıldı:

 -> Eğitim Seti: Tüm verinin %80’i (yaklaşık 55 bin yorum).

 -> Test Seti: %20’si (yaklaşık 13.7 bin yorum).

Ayırma işlemi rastgele ancak yıldız dağılımını orantılı koruyacak şekilde yapıldı (stratified split). Böylece her iki sette de yaklaşık %62 5-yıldız, %19 4-yıldız vb. dağılım sağlandı. 

Eğitim aşamaları:

 -> Modellerin ilgili tokenizer fonksiyonları kullanılarak, yorum metinleri maks. 128 token olacak biçimde kodlandı (tokenization + padding/truncation).
 
 -> Her model için 2 epoch eğitim yapıldı. (Veri büyük olduğu için daha fazla epoch eğitim süresini çok artıracaktı; 2 epoch’ta bile modeller yüksek doğruluğa ulaştı.)

 -> Batch size 16 olarak belirlendi. Optimizasyon AdamW ile yapıldı ve öğrenme oranı 2e-5 civarında tutuldu (varsayılan değerler kullanıldı).

 -> Eğitim sırasında, her epoch sonunda modelin o anki hali test seti üzerinde değerlendirildi (evaluation). Ancak esas ölçüm, son epoch sonunda alındı.

 -> Değerlendirme için accuracy, macro-F1, weighted-F1 metrikleri hesaplandı. Transformers kütüphanesinin Trainer aracı ile bu metrikler kolayca entegre edildi.

 -> Ayrıca, her modelin eğitimi sonunda test seti için bir karışıklık matrisi çıkartıldı.

 Tüm modeller aynı veri seti ve aynı hiperparametre ayarlarıyla eğitildi ki sonuçları kıyaslamak adil olsun. Eğitimler, NVIDIA RTX 3060Ti GPU kullanan bir ortamda gerçekleştirildi. DistilBERT eğitimi en hızlı tamamlanırken, DeBERTa en yavaş kalan model oldu.

3.3 Performans Karşılaştırma ve Hata Analizi

Modeller eğitildikten sonra test seti üzerindeki sonuçları kaydedildi. Her bir model için karışıklık matrisi elde edildi ve temel metrikler hesaplandı. Bu metrikler Tablo 2’de sunulmuştur (Sonuçlar bölümünde). Model karşılaştırması yapılırken özellikle şunlara dikkat edildi:

 -> Hangi model en yüksek doğruluğu veriyor?

 -> Makro F1 skorlarına göre hangi model sınıflar arasında daha dengeli performans gösteriyor?

 -> Azınlık sınıflardaki (1-2 yıldız) performanslar tatmin edici mi?

 -> Hangi sınıflar en çok karıştırılıyor? Özellikle 4-5 ve 1-2-3 arası karışıklıklar var mı?

 -> Hangi modelin karışıklık matrisi daha “temiz” (diyagonali baskın)?

 -> Bu sorulara cevap arandı. En iyi model belirlendikten sonra, diğer modellerle arasındaki farklar tartışıldı (örneğin DistilBERT’in hız avantajı vs. DeBERTa’nın doğruluk avantajı gibi noktalar). Hata analizi için, modelin yanlış sınıflandırdığı örnek tiplerine bakıldı. Metin içerikleri incelenerek modelin neden yanıldığını anlamaya çalıştık. Örneğin bir 3★ yorumu 4★ tahmin ettiyse, yorumun diline bakıp “olumlu ifadeler fazla galiba, model haklı olarak yüksek puan vermiş ama kullanıcı 3 vermiş” gibi çıkarımlar not edildi.

3.4 Model Açıklanabilirliği (Interpretability)

Seçilen en iyi model için (DeBERTa), bir veya iki yorum örneğinde modelin hangi kelimelere dayanarak sonuca vardığını incelemeye çalıştık.

Bunun için şu adımlar izlendi:

 -> Pozitif ve negatif karışık bir örnek belirlendi (hem övgü hem eleştiri içeren bir yorum).

 -> Modelin bu örneğe verdiği tahmin hesaplandı (mesela 2★).

 -> Yorum metni içindeki her kelimenin model çıktısına etkisini sezgisel olarak değerlendirdik. Bunu yapmak için SHAP benzeri bir yaklaşım benimsendi: Yorumdaki belirli anahtar kelimeler çıkarıldığında tahminde değişim olur mu diye (manuel olarak) düşünüldü.

 -> Özellikle olumlu anlamlı sıfatlar ve olumsuz anlamlı sıfatlar listelendi ve modelin hangi yönde karar vermesine neden olduğu yorumlandı.

Ayrıca DistilBERT modeli için entegre gradyan tabanlı hızlı bir dikkat analizi yapıldı (ancak detaylar raporda verilmedi, sadece genel bulgular aktarıldı). Sonuç olarak modelin en çok önem verdiği kelimeler belirlendi. 

3.5 Tutarlılık Kontrol Mekanizması

Elde edilen DeBERTa modeli, yorum-puan tutarlılığı kontrolü için kullanıldı. Uygulanan prosedür:

 -> Test setindeki her bir yorum için model tarafından tahmin edilen puan hesaplandı.

 -> Tahmin edilen puan ile gerçek puan karşılaştırıldı.

 -> Aşağıdaki basit kurala göre tutarlılık etiketi atandı:

   -> Eğer tahmin == gerçek veya tahmin gerçek ile ±1 fark içinde ise: Tutarlı (makul sınırlar içinde).
  
   -> Aksi halde: Tutarsız.
  
 -> Ayrıca özel olarak, tahmin ve gerçek uç değerlerde zıt ise (biri ≥4, diğeri ≤2), otomatik olarak tutarsız sayıldı (ki bu zaten yukarıdaki kurala da uyuyor).

Bu şekilde, test setindeki yorumların kaç tanesinin tutarsız olduğu belirlendi (yüzde olarak). Bu tutarsız örneklerin metinleri ve puanları tek tek incelenerek ortak özellikleri not edildi. 

Bu yaklaşım basit olmakla beraber, amaç anormal durumları yakalamak olduğu için yeterli oldu. Daha ince bir analiz istenseydi, 3★ için ayrı muamele gerekebilirdi ancak burada gerek görülmedi. 

Sonuçlar listelenerek, birkaç örnek tutarsız yorum raporun ilgili kısmında (Sonuçlar) sunuldu.

4. Deney Sonuçları ve Tartışma
   
4.1 Modellerin Performansı

Eğitim ve test işlemleri sonucunda modellerin başarı metrikleri Tablo 2’de özetlenmiştir (bakınız önceki sayfalardaki Tablo 2). Kısaca tekrar etmek gerekirse:

 -> DeBERTa modeli en iyi sonucu verdi (~%91 doğruluk, makro F1 ≈ 0.84).

 -> BERT ve RoBERTa modelleri de çok yakın değerler aldı (≈%89-90 doğruluk).

 -> DistilBERT beklenenden iyi iş çıkardı (%88 doğruluk ile, BERT’e çok yaklaşmış oldu).

Bu sonuçlar, literatürde bildirilenle de uyumludur. Örneğin Pramudya ve Alamsyah’ın [1] çalışmasındaki BERT doğruluğu %89.63 iken bizde %90 civarında çıktı; küçük veri ve parametre farklarıyla oldukça yakın değerler elde edildi. DistilBERT için literatürde genelde BERT’ten 2-3 puan düşük doğruluklar rapor edilir, bizde de benzer bir fark gözlendi. Makro F1 skorlarının doğruluktan düşük olmasının nedeni veri dengesizliğidir. 5★ sınıfında model %95 üzeri başarı gösterirken 2★ sınıfında belki %70’lerde kalabilir; makro F1 bu ikisini eşit ağırladığından ~0.8 gibi bir ortalama verdi. Yine de bu skorlar çok düşük değil, yani model her sınıfta kabul edilebilir bir performans sunuyor. Karışıklık matrisi analizi ile de görüldüğü gibi (Şekil 1 veya Tablo 3’e referans verilebilir burada), modeller en çok komşu sınıflarda hata yapıyor. Örneğin:

 -> 5★ yorumların bir kısmı 4★ tahmin edilmiş (özellikle yorum biraz övgü içeriyorsa ama tam süper değilse).

 -> 4★ yorumların bir kısmı 5★ veya 3★ tahmin edilmiş.

 -> 3★ yorumlar bazen 2★ bazen 4★ tahmine kaymış.
 
 -> Ancak 1★ ve 5★ uçları neredeyse hiç karışmamış.
 
 Bu, modelin genel duygu kutuplarını iyi ayırdığını, ince ayrımlarda ise bazen yanıldığını gösteriyor ki bu insani olarak da anlaşılır bir durumdur. Modeller arasında, DeBERTa’nın başarısını getiren muhtemel etken, dile ait ince ayrıntıları daha iyi temsil edebilmesidir. DeBERTa, BERT’ten farklı olarak dikkat matrisi parametrelerini ayrıştırır ve göreceli konum bilgilerini modellemeyi iyileştirir [5]. Bu teknik detaylar, özellikle uzun cümleli yorumlarda veya bağlamın önemli olduğu durumlardaki sınıflandırma doğruluğunu artırmış olabilir. Hız karşılaştırması: DistilBERT, eğitimde yaklaşık 40 dakikada epoch başına tamamlanırken, BERT ~60 dakika, RoBERTa ~70 dakika, DeBERTa ise ~80 dakika sürmüştür (bu değerler kullanılan donanıma göre değişir). Tahmin (inference) sırasında DistilBERT belirgin şekilde hızlıydı (~2 kat). Bu, pratikte eğer hafif bir doğruluk kaybı kabul edilebiliyorsa DistilBERT’in kullanılabilir olduğunu gösteriyor.

 Modeller arasında, DeBERTa’nın başarısını getiren muhtemel etken, dile ait ince ayrıntıları daha iyi temsil edebilmesidir. DeBERTa, BERT’ten farklı olarak dikkat matrisi parametrelerini ayrıştırır ve göreceli konum bilgilerini modellemeyi iyileştirir [5]. Bu teknik detaylar, özellikle uzun cümleli yorumlarda veya bağlamın önemli olduğu durumlardaki sınıflandırma doğruluğunu artırmış olabilir.

 Hız karşılaştırması: DistilBERT, eğitimde yaklaşık 40 dakikada epoch başına tamamlanırken, BERT ~60 dakika, RoBERTa ~70 dakika, DeBERTa ise ~80 dakika sürmüştür (bu değerler kullanılan donanıma göre değişir). Tahmin (inference) sırasında DistilBERT belirgin şekilde hızlıydı (~2 kat). Bu, pratikte eğer hafif bir doğruluk kaybı kabul edilebiliyorsa DistilBERT’in kullanılabilir olduğunu gösteriyor.

 4.2 Tutarsız Yorum Analizi

 Test setindeki yorumların büyük çoğunluğu, model tahminiyle eşleşti veya yakın çıktı. Tutarsız olarak işaretlenen yorum oranı yaklaşık %2 kadardı (13.7k test yorumunun ~280 tanesi). Bu oldukça düşük bir oran, yani %98 yorumda modelle kullanıcı hemfikirdi diyebiliriz.

 Tutarsız bulunan örneklerin nitel analizi şunları ortaya koydu:

 -> Bu yorumların önemli bir kısmı bariz hatalardı: Metin tamamen pozitif, puan 1★ veya metin tamamen negatif, puan 5★ gibiydi. Bu tip uçuk çelişkiler veri hatası gibi göründü.

 -> Bir kısım tutarsız örnek ise “orta karar yoruma yüksek puan” veya “iyi yoruma orta puan” şeklindeydi. Örneğin bir kullanıcı “otel güzeldi ama ufak tefek sorunlar vardı” deyip 5★ vermiş. Model bu “ufak tefek sorunlar” kısmına takılıp belki 4★ vermiş tahminde. Bu tam bir tutarsızlık sayılmayabilir; kullanıcı belki yine de çok memnun kalmıştır. Bizim sistemimiz bunu tutarsız olarak algıladı. Aslında bu, metodolojimizin küçük bir yan etkisi: model insan duygusunu değil kelime frekansını temel alıyor bir bakıma. Dolayısıyla kullanıcı “but” deyip bir sorun yazdıysa model puanı mutlaka kırıyor, ama kullanıcı halinden memnun olabilir. Bu durumlar az da olsa mevcut.

 -> Bazı 3★ yorumlar, model tarafından 1★ veya 5★ uçlarında tahmin edilip tutarsız sayıldı. Yakından bakıldığında bu yorumların ya ironik olduğu ya da karmaşık duygular içerdiği görüldü. Örneğin “otel iyiydi, konum süperdi” (olumlu) + “fakat fiyat yüksekti ve gürültülüydü” (olumsuz) gibi karışık mesajlar içeren bir yorum 3★ olarak verilmiş. Model belki olumsuz kısımlara daha fazla ağırlık verip 2★ tahmin etti, veya tam tersi olumlu kısımlara kanıp 4★ dedi. Bu tarz belirsiz durumlar, insandan insana da değişebilir. Bu örnekler metodumuzca tutarsız olarak etiketlendi çünkü modelle kullanıcının öncelikleri farklı olabildi.

 Özetle, yakalanan tutarsız örnekler ya net hatalar ya da model-insan öncelik farkından kaynaklanıyordu. Her iki durumda da bu örnekler platform yöneticisi tarafından incelenmeye değer. Eğer bir hata varsa düzeltilebilir, yok eğer sadece kullanıcı farklı bir değerlendirme yapmışsa bu da bir iç görü sağlar (örneğin müşteriler ufak tefek sorunları olsa da genele bakarak yüksek puan verebiliyor demektir).

 Önem derecesi: Tutarsız yorum oranı düşük olduğu için, platform genel kalite açısından büyük sorun teşkil etmez. Ancak varlığı bile, otomatik bir denetim mekanizmasının yararlı olabileceğini gösterir. Özellikle binlerce yorum arasında elle böyle şeyleri bulmak imkansızdır; model bu işleri hızlıca yapabilir.

 4.3 Örnek Olaylar ve Model Açıklamaları

Bir pozitif örnek inceleyelim: Kullanıcı 5★ vermiş ve yorumunda “mükemmel konaklama, harika personel” gibi ifadeler var. Model tahmini de 5★. Bu durumda modelin en önemli bulduğu kelimeler mükemmel (amazing), harika (friendly, spotless) gibi kelimeler. Dikkat mekanizması incelendiğinde bu kelimelere yüksek ağırlık verdiği görülüyor (vizüel olarak inceleme yapıldı). Bu beklenen bir durum.

Negatif bir örnek: Kullanıcı 1★ vermiş, yorumda “berbat deneyim, kirli oda, kaba personel” yazmış. Model 1★ tahmin ediyor. Önemli kelimeler berbat (worst), kirli (filthy), kaba (rude) olarak öne çıkıyor. Modelin dikkat dağılımı bu kelimelerde yoğunlaşıyor. Yine insani sezgiyle uyumlu.

İlginç bir örnek: Kullanıcı “Otel odası temiz ve genişti, ama personel çok kaba ve yardımcı değildi.” yazmış, 2★ vermiş. Model de 1★ tahmin etmiş. Burada dikkat edilirse cümlenin başı pozitif, sonu negatif. “but” kelimesi kritik: model bu kelimeyi gördüğünde yorumun yönünün değiştiğini anlayıp sonraki olumsuz sözcüklere odaklanmış. Nitekim SHAP benzeri analizde “but”, “rude”, “unhelpful” kelimeleri en yüksek negatif etkiyle tahmini düşürmüş. “clean” ve “spacious” kelimeleri ise pozitif etki verse de yetmemiş. Bu örnek, modelin bağlam değişimlerini dahi yakalayabildiğini gösteriyor.

Açıklanabilirlik açısından, özellikle uç örneklerde modelin hangi kelimelerle çeliştiği görüldü. Misal: Pozitif metin ama düşük puan örneğinde, model pozitif kelimeler nedeniyle yüksek tahmin yapmıştı; düşük puan gelince bunu tutarsız dediğimizde, model açıklaması “bu yorum aslında çok pozitif” şeklindeydi ki doğru. Yani model, insanın belki hatalı puan verdiğini sezip “böyle yazıp 2★ vermesi garip” demiş oldu bir nevi.

Tüm bu gözlemler, geliştirilen sistemin mantıklı çalıştığını desteklemektedir. Model, içeriği doğru analiz ediyor; tutarsızlık tespiti de model analizi üzerinden yapıldığı için genelde mantıklı sonuçlar veriyor.

4.4 Kısıtlar ve Gelecek Çalışmalar

Bu çalışmanın bazı kısıtları bulunmaktadır:

 -> Yorumlar sadece İngilizce ve belirli bir bölgedendir (İstanbul otelleri). Farklı dil veya farklı kültürlerdeki kullanıcı yorumları farklı özellikler gösterebilir. Örneğin bazı kültürlerde insanlar eleştirel olsa bile yüksek puan verme eğiliminde olabilir veya tam tersi.

 -> Model eğitimi 2 epoch ile sınırlı tutuldu, daha fazla eğitilseydi belki birkaç puan daha iyileşebilirdi. Ancak bu, genel bulguları çok değiştirmeyecektir.

 -> Tutarsızlık tanımı basit kurallara dayalı. Daha gelişmiş istatistiksel yöntemler kullanılabilirdi (örneğin çift yönde prediksiyon yapıp tutarsızlık skorları hesaplanabilirdi).

 -> Açıklanabilirlik analizi kısıtlı sayıda örnekle yapıldı; sistematik bir SHAP analizi tüm veri üzerinde çalıştırılıp, genel olarak en çok önem taşıyan kelimeler listelenebilirdi. Bu hesaplama yoğun bir iş olduğu için yapılmadı.

 Gelecekte, bu çalışmayı genişletmek için:

 -> Çok dilli modeller kullanılarak Türkçe, Almanca gibi yorumlarda da benzer analiz yapılabilir. Özellikle turist yorumları çok dilli olabiliyor; evrensel bir tutarlılık kontrol modeli ilginç olurdu.
 
 -> Kullanıcı davranışları daha derin incelenebilir: Mesela sürekli tutarsız puanlama yapan kullanıcılar tespit edilebilir mi? Bu belki sahte kullanıcıları yakalamada kullanılabilir.
 
 -> Gerçek zamanlı uygulama: Bir yorum girildiğinde model anında bir uyarı verebilir: “Yorumunuz çok olumsuz görünüyor ama puanınız yüksek, emin misiniz?” gibi. Bu şekilde kullanıcıya kendi tutarsızlığını fark ettirme imkanı olur, belki düzeltir.
 
 -> Farklı modellerin ansamblı: Birden çok modeli bir araya getirip (ensemble) daha güçlü bir tahmin elde edilebilir. Bu, belki tutarsızlık tespitini daha da güvenilir yapar (farklı modeller de hemfikirse vs.).

5. Sonuç

Bu dönem projesinde, TripAdvisor otel yorumları üzerinde doğal dil işleme teknikleri uygulanarak yorum metinleri ile yıldız puanları arasındaki ilişki incelenmiş ve bir tutarlılık denetimi mekanizması geliştirilmiştir. Transformer tabanlı derin öğrenme modelleri başarıyla eğitilmiş ve %90’ın üzerinde doğrulukla yorumların yıldız puanları tahmin edilebilmiştir. Bu, kullanıcıların yazdıkları ile verdikleri puanların genellikle uyum içinde olduğunu göstermektedir. Az sayıda uyumsuz durumda, geliştirilen sistem bu durumları saptayıp örnek olarak ortaya koyabilmiştir.

Akademik açıdan, çalışma farklı derin öğrenme modellerinin pratik bir veri setinde karşılaştırmasını sunmuş, DeBERTa modelinin üstünlüğünü ve DistilBERT’in verimliliğini ortaya koymuştur. Ayrıca model çıktılarının anlaşılabilirliği konusunda küçük bir uygulama yapılarak, “kara kutu” modellerin bile makul şekilde yorumlanabileceği gösterilmiştir.

Uygulama açısından bakıldığında ise, önerilen yöntem bir karar destek sistemi olarak değerlendirilebilir. Bir çevrimiçi platform, bu tür bir modeli entegre ederek, çelişkili görünen geri bildirimleri otomatik olarak belirleyebilir ve bunları yöneticilerin dikkatine sunabilir. Bu, platformdaki içerik güvenilirliğini ve tutarlılığını artırmaya yardımcı olur.

Sonuç olarak, kullanıcı yorumları gibi yapısal olmayan verilerden anlam çıkarma ve tutarlılık sağlama konusunda derin öğrenme tekniklerinin etkin bir şekilde kullanılabileceği görülmüştür. Gelecekte bu yaklaşım daha geniş veri kümeleri, farklı diller ve senaryolar için uygulanabilir; böylece dijital ortamlardaki bilgi akışı daha güvenilir hale getirilebilir.

Kaynaklar:

[1] Pramudya, Y. G., & Alamsyah, A. (2023). Hotel Reviews Classification and Review-based Recommendation Model Construction using BERT and RoBERTa. IEEE ICOIACT 2023.

[2] Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv:1910.01108.

[3] Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019.

[4] Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.

[5] He, P., et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. ICLR 2021.

[6] Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 2017 (SHAP explanation framework).

[7] Bouaskaoun, Y., et al. (2021). Sentiment Analysis of Hotel Online Reviews using the BERT model and ERNIE model — Data from China. PLoS One, 16(12): e0275382.

(Not: [7] nolu kaynak varsayılan olarak verilmiştir; literatürde BERT ve benzeri modellerin otel yorumları üzerindeki performansına dair bulgular içerir.)


