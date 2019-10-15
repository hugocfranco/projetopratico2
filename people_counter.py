# Formas de uso:

# Para ler e gravar novamente em vídeo:
''' python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 \
	--output output/output_01.avi '''

# Para ler da webcam e gravar novamente em disco:
''' python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
	--output output/webcam_output.avi  '''

# importar os pacotes necessários

# constrói os argumentos de analise e analisa os argumentos
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
                help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
                help="# of skip frames between detections")
args = vars(ap.parse_args())

# inicializa uma lista de classes de rótulos que o MobileNet SSD foi treinado para detectar
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# carregar nosso modelo serializado do disco
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Se um caminho de vídeo não foi fornecido, pegue uma referência à webcam
if not args.get("input", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

# caso contrário, pegue uma referência ao arquivo de vídeo
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])

# inicialize o gravador de vídeo (instanciamos mais tarde, se necessário)
writer = None

# inicialize as dimensões do quadro (as definiremos assim que lermos o primeiro quadro do vídeo)
W = None
H = None

# instanciar nosso rastreador centróide e, em seguida, inicializa
# uma lista para armazenar cada um de nossos rastreadores de
# correlação dlib, seguidos por um dicionário para mapear cada único ID
# de objeto para um TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# inicializa o número total de quadros processados ​​até o momento, juntamente
# com o número total de objetos que foram movidos para cima ou para baixo
totalFrames = 0
totalDown = 0
totalUp = 0

# inicia o estimador de taxa de transferência de quadros por segundo
fps = FPS().start()

# loop sobre quadros do vídeo
while True:
    # utiliza o próximo quadro e manipula de acordo se estiver lendo
    # um video da webcam ou um arquivo de vídeo
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame

    # se estamos vendo um vídeo e não capturamos um quadro, chegamos ao final do vídeo
    if args["input"] is not None and frame is None:
        break

    # redimensiona o quadro para ter uma largura máxima
    # de 500 pixels (quanto menos dados tivermos, mais
    # rápido poderemos processá-lo) e depois converta
    # o quadro de BGR para RGB para dlib
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Se a dimenção do quadro está vazia, então definimos
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # Se passarmos o parêmetro para gravar o vídeo no disco, então inicializamos o gravador
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                 (W, H), True)

    # inicializa o status atual junto com nossa lista
    # de caixas retangulares delimitadoras retornadas por:
    # (1) nosso detector de objetos ou
    # (2) pelos rastreadores de correlação
    status = "Waiting"
    rects = []

    # verifica se devemos executar um método que usa mais recursos computacionais para ajudar nosso rastreador
    if totalFrames % args["skip_frames"] == 0:
        # define o status e inicializa o nosso novo conjunto de rastreadores de objetos
        status = "Detecting"
        trackers = []

        # converte o quadro em um blob e passe o blob pela rede e obtenha as detecções

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # loop sobre as detecções
        for i in np.arange(0, detections.shape[2]):
            # extrai a confiança (ou seja, a probabilidade) associada à previsão
            confidence = detections[0, 0, i, 2]

            # filtra as detecções fracas, exigindo uma confiança mínima
            if confidence > args["confidence"]:
                # extrai o índice da classe de rótulos da lista de detecções
                idx = int(detections[0, 0, i, 1])

                # se a classe de rótulo não for uma pessoa, ignore-a
                if CLASSES[idx] != "person":
                    continue

                # calcula as coordenadas (x, y) da caixa delimitadora para o objeto
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # constroi um objeto retângular dlib a partir das coordenadas da caixa delimitadora e inicia o rastreador de correlação dlib
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # adiciona o rastreador à nossa lista de rastreadores para que possamos utilizá-lo durante o pulo de quadros
                trackers.append(tracker)

    # caso contrário, devemos utilizar nosso objeto *trackers*
    # em vez de objetos *detectors* para obter uma maior taxa
    # de transferência de processamento de quadros
    else:
        # loop sobre os rastreadores
        for tracker in trackers:
            # define o status do nosso sistema como 'Tracking' em vez de 'Waiting' ou 'Detecting'
            status = "Tracking"

            # atualiza o rastreador e pega a posição atualizada
            tracker.update(rgb)
            pos = tracker.get_position()

            # descompacta o objeto de posição
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # adiciona as coordenadas da caixa delimitadora à lista de retângulos
            rects.append((startX, startY, endX, endY))

    # desenha uma linha horizontal no centro do quadro,
    # assim que um objeto cruzar essa linha,
    # determinaremos se eles estavam se movendo 'para cima' ou 'para baixo'
    cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

    # usa o rastreador de centróide para associar os
    # (1) centroides de objetos antigos com
    # (2) os centróides de objetos recém-calculados
    objects = ct.update(rects)

    # loop sobre os objetos rastreados
    for (objectID, centroid) in objects.items():
        # verifica se existe um objeto rastreável para o ID do objeto atual
        to = trackableObjects.get(objectID, None)

        # Se não houver um objeto rastreável, cria um
        if to is None:
            to = TrackableObject(objectID, centroid)

        # caso contrário, há um objeto rastreável para que possamos utilizá-lo para determinar a direção
        else:
            # a diferença entre a coordenada y do centróide *atual*
            # e a média dos centróides *anteriores* nos dirá em que
            # direção o objeto está se movendo
            # (negativo para 'para cima' e positivo para 'para baixo')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # verifica se o objeto foi contado ou não
            if not to.counted:
                # se a direção for negativa (indicando que o objeto está se movendo para cima)
                # *E* o centróide estiver acima da linha central, conte o objeto
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    to.counted = True

                # se a direção for positiva (indicando que o objeto está se movendo para baixo)
                # *E* o centróide estiver abaixo da linha central, conte o objeto
                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    to.counted = True

        # armazena o objeto rastreável em nosso dicionário
        trackableObjects[objectID] = to

        # desenha o ID do objeto e o centróide do objeto no quadro de saída
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # constroi uma tupla de informações que exibiremos no quadro
    info = [
        ("Entrada", totalUp),
        ("Saidas", totalDown),
        ("Status", status),
    ]

    # faz um loop sobre as tuplas de informações e desenha em nosso quadro
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # cverifica se devemos gravar o quadro no disco
    if writer is not None:
        writer.write(frame)

    # mostra o quadro de saída
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # se a tecla 'q' for pressionada, interrompe o loop
    if key == ord("q"):
        break

    # incrementa o número total de quadros processados até o momento e atualiza o contador do FPS
    totalFrames += 1
    fps.update()

# para o cronômetro e exiba informações de FPS
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# verifica se precisamos liberar o ponteiro do gravador de vídeo
if writer is not None:
    writer.release()

# se não estivermos usando um arquivo de vídeo, para o fluxo de vídeo da câmera
if not args.get("input", False):
    vs.stop()

# caso contrário, solta o ponteiro do arquivo de vídeo
else:
    vs.release()

# fecha todas as janelas abertas
cv2.destroyAllWindows()
