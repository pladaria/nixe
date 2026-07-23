# IPC, HIPC y CMIF en Nintendo Switch 1

Este documento explica el sistema de comunicación entre procesos de Horizon,
las funciones de HIPC y CMIF, su relación con sesiones, puertos, handles,
servicios y memoria, y la forma en que Nixe implementa actualmente esas capas.

La información del protocolo procede de documentación pública y de ingeniería
inversa de la comunidad. Los nombres de algunos campos no son nombres oficiales
publicados por Nintendo.

> **Estado de la implementación de Nixe**
>
> La implementación descrita aquí es funcional, está validada con pruebas y
> tiene límites defensivos explícitos, pero todavía es parcial y está en
> evolución. La arquitectura, los servicios disponibles, los comandos
> implementados, los límites, los códigos de error y algunos detalles de la
> codificación podrían cambiar a medida que avance la implementación. Este
> documento describe el estado del repositorio en el momento de escribirlo, no
> una interfaz estable que el resto del proyecto deba considerar congelada.

## Resumen en una frase

Una aplicación no llama directamente a una función Rust de un servicio:
construye un mensaje HIPC/CMIF en memoria, ejecuta una SVC con el handle de una
sesión y recibe en el mismo búfer una respuesta que puede contener datos,
handles u objetos de dominio.

```text
Aplicación invitada
    │
    │ escribe un mensaje en el command buffer del TLS
    │
    │ HIPC: transporte, descriptores, PID y handles
    │ CMIF: objeto, command ID, argumentos y result
    ▼
svcSendSyncRequest(handle)
    │
    ▼
Kernel de Horizon / dispatcher de SVC de Nixe
    │
    ├── sesión hacia un servicio emulado por Nixe
    │       └── decodifica, valida y ejecuta la operación
    │
    └── sesión genérica hacia otro proceso invitado
            └── entrega la petición al endpoint servidor
    │
    ▼
respuesta HIPC/CMIF en el command buffer
```

Los términos esenciales son:

| Término | Papel |
| --- | --- |
| IPC | El concepto general de comunicación entre procesos. |
| HIPC | El formato de transporte de Horizon: cabecera, handles, descriptores de búfer y sección de datos. |
| CMIF | El protocolo habitual de comandos y objetos que se coloca dentro de la sección de datos de HIPC. |
| SVC | La entrada desde código invitado al kernel para conectar, enviar, esperar, responder o cerrar. |
| Puerto | Punto de publicación y creación de sesiones. |
| Sesión | Canal con endpoint cliente y servidor por el que se intercambian peticiones síncronas. |
| Handle | Identificador local a un proceso para una sesión u otro objeto del kernel. |
| Servicio | Interfaz nombrada, por ejemplo `sm:`, `fsp-srv` o `hid`, que acepta comandos. |
| Dominio CMIF | Multiplexación de varios objetos de servicio mediante IDs sobre una sola sesión/handle. |

## Las capas y por qué no deben confundirse

IPC no es un único formato. En una llamada normal intervienen varias capas:

```text
┌──────────────────────────────────────────────────────────────┐
│ Semántica del servicio                                      │
│ Ej.: IFile::Read(offset, size), GetOperationMode()           │
├──────────────────────────────────────────────────────────────┤
│ CMIF                                                         │
│ command ID, token, result, argumentos y objetos de dominio   │
├──────────────────────────────────────────────────────────────┤
│ HIPC                                                         │
│ tipo, handles, PID, descriptores de memoria y raw data       │
├──────────────────────────────────────────────────────────────┤
│ Sesión y objetos del kernel                                  │
│ endpoints cliente/servidor, cola, espera, respuesta, cierre  │
├──────────────────────────────────────────────────────────────┤
│ SVC + CPU + memoria virtual                                  │
│ registros, TLS o user buffer y acceso a memoria invitada     │
└──────────────────────────────────────────────────────────────┘
```

Cada capa responde a una pregunta distinta:

- HIPC dice **cómo transportar** los componentes del mensaje.
- CMIF dice **qué objeto y comando** se invocan y cómo se expresa el resultado.
- El servicio dice **qué significa** ese comando.
- La sesión y el kernel dicen **quién habla con quién** y cuándo se bloquea o
  despierta un hilo.
- El handle permite que el proceso se refiera al objeto sin conocer un puntero
  del kernel o del emulador.

CMIF no reemplaza a HIPC: normalmente viaja dentro de HIPC. HIPC tampoco
implica necesariamente CMIF. Existe además TIPC, un protocolo más pequeño
introducido para algunos servicios en versiones posteriores de Horizon, que
también usa el transporte HIPC. Nixe implementa actualmente el camino CMIF
descrito en este documento, no una implementación general de TIPC.

## Puertos, sesiones, servicios y handles

### Puerto y sesión

Un puerto es un punto de conexión. Un servidor publica o administra un puerto y
los clientes se conectan a él. Cada conexión crea una sesión con dos extremos:

```text
                 puerto "ejemplo"
                       │ connect
                       ▼
Proceso cliente   endpoint cliente ═════ endpoint servidor   Proceso servidor
        handle C ───────┘                     └─────── handle S
```

Los números de handle no tienen por qué coincidir. Cada tabla de handles es
local al proceso. Ambos handles identifican endpoints relacionados, pero no son
el mismo entero global.

Una petición síncrona sigue este ciclo:

1. El cliente envía una petición por su endpoint.
2. Si aún no hay respuesta, su hilo queda suspendido.
3. El servidor espera con `ReplyAndReceive`, acepta la petición y la procesa.
4. El servidor responde.
5. El cliente se despierta y continúa con el resultado.

La sesión también conserva el estado de cierre de ambos extremos. Cerrar el
último handle de un endpoint hace visible el cierre al peer.

### El Service Manager (`sm:`)

Las aplicaciones no suelen conocer el puerto de cada servicio. Primero llaman
a `ConnectToNamedPort("sm:")`. El resultado es una sesión con el Service
Manager. Sobre ella realizan:

```text
ConnectToNamedPort("sm:")
        │
        ▼
handle de sesión sm:
        │ CMIF command 0: RegisterClient, con PID
        ▼
cliente registrado
        │ CMIF command 1: GetService("fsp-srv")
        ▼
handle de sesión fsp-srv
```

`sm:` es por tanto un registro y broker de servicios, no el transporte IPC.
La petición a `sm:` ya es una petición HIPC/CMIF.

### Handles copiados y movidos

HIPC puede incluir dos listas de handles:

- **copy handles**: el receptor obtiene otra referencia al mismo objeto; el
  remitente conserva la suya;
- **move handles**: la propiedad se transfiere; el handle de origen se consume.

Esto importa para objetos devueltos por un servicio. Por ejemplo, abrir un
filesystem o un fichero fuera de un dominio devuelve normalmente un handle
movido a una nueva sesión/objeto. Un evento o memoria compartida suele
devolverse como handle copiado según la ABI concreta del comando.

Los valores especiales `0xffff8000` y `0xffff8001` representan,
respectivamente, el hilo y el proceso actuales en operaciones que aceptan esos
pseudo-handles. No son entradas ordinarias de la tabla de handles.

## El command buffer y el TLS

En la ruta habitual `SendSyncRequest`, el mensaje está en el command buffer de
la región local del hilo, el TLS. El registro de sistema de TLS identifica esa
región:

| Modo de CPU | Registro usado por Nixe |
| --- | --- |
| AArch64 | `TPIDR_EL0` |
| AArch32 | `TPIDRURW` |

Nixe crea y mapea la región TLS al construir el proceso, inicializa el registro
correspondiente y usa sus primeros `0x100` bytes como command buffer IPC.
El búfer pertenece al espacio virtual invitado: sus direcciones y las de sus
descriptores nunca son punteros Rust.

```text
TLS del hilo invitado
┌───────────────────────────────┐  ← TPIDR_EL0 / TPIDRURW
│ command buffer HIPC (0x100)   │
├───────────────────────────────┤
│ resto de datos locales        │
└───────────────────────────────┘
```

`SendSyncRequestWithUserBuffer` usa en cambio una dirección y un tamaño
explícitos. Nixe valida que ambos respeten la alineación de página y que el
rango no sea vacío ni desborde. La ruta de sesiones genéricas transporta ese
búfer completo.

> **Limitación actual:** el codec de servicios incorporados sigue limitado al
> command buffer TLS de `0x100` bytes. Por ello, la variante con user buffer no
> es todavía una ruta utilizable de forma general contra esos servicios: el
> tamaño exigido por la ABI de la SVC es mayor que el aceptado por ese codec.
> Esta frontera deberá separarse o generalizarse; es una parte inestable de la
> implementación.

## HIPC: la capa de transporte

HIPC organiza metadatos y referencias a memoria alrededor de una sección de
datos. Su disposición conceptual es:

```text
offset variable
┌────────────────────────────────────────────┐
│ HeaderData: 2 palabras de 32 bits          │
├────────────────────────────────────────────┤
│ SpecialHeaderData, si existe               │
│   ├── PID opcional                         │
│   ├── copy handles                         │
│   └── move handles                         │
├────────────────────────────────────────────┤
│ send-static descriptors                    │
├────────────────────────────────────────────┤
│ send / receive / exchange buffer descs.    │
├────────────────────────────────────────────┤
│ raw data words                             │
│   └── CMIF alineado a 16 bytes             │
├────────────────────────────────────────────┤
│ receive-static descriptors, si existen     │
└────────────────────────────────────────────┘
```

### Cabecera HIPC

Nixe interpreta las dos palabras iniciales así:

| Campo | Bits | Significado |
| --- | ---: | --- |
| `command_type` | palabra 0, 0..15 | Tipo de mensaje que después interpreta CMIF. |
| send-static count | palabra 0, 16..19 | Número de descriptores estáticos de entrada. |
| send-buffer count | palabra 0, 20..23 | Número de buffers enviados. |
| receive-buffer count | palabra 0, 24..27 | Número de buffers de salida. |
| exchange-buffer count | palabra 0, 28..31 | Número de buffers bidireccionales. |
| data-word count | palabra 1, 0..9 | Tamaño de la sección raw en palabras de 32 bits. |
| receive-static mode | palabra 1, 10..13 | Ninguno, automático o número codificado de entradas. |
| special-header present | palabra 1, bit 31 | Indica PID y/o listas de handles. |

Los bits reservados deben ser cero. Los contadores se validan antes de reservar
o recorrer memoria. El formato público también define un
`ReceiveListOffset` en los bits 20..30 de la segunda palabra. El codec actual
no lo modela como campo independiente y espera la lista después de los raw
data; ampliar esta parte será necesario para aceptar todas las variantes
válidas del transporte.

### Cabecera especial

Cuando existe, indica:

- si se envía el PID del proceso;
- cuántos handles se copian;
- cuántos handles se mueven.

El PID no es un argumento de usuario ordinario. El kernel lo rellena o valida
como identidad del emisor; numerosos comandos, como `sm:RegisterClient`,
`fsp-srv:SetCurrentProcess`, `hid:CreateAppletResource` y la apertura de
`appletOE`, requieren que esté presente.

### Descriptores de memoria

Los argumentos grandes no caben en el command buffer. HIPC transporta
descriptores que contienen dirección invitada, tamaño y, cuando corresponde,
modo de mapeo:

| Descriptor | Uso conceptual |
| --- | --- |
| send static / pointer | Región pequeña que el servidor lee mediante un puntero. |
| send buffer | Región que el cliente entrega como entrada. |
| receive buffer | Región en la que el servidor escribe la salida. |
| exchange buffer | Región utilizable en ambas direcciones. |
| receive static / pointer | Región de salida descrita fuera de la tabla principal. |

Los buffers mapeables llevan un modo `Normal`, `NonSecure`, `Invalid` o
`NonDevice`. El formato divide direcciones y tamaños entre varios campos de
bits; el codec de Nixe los reconstruye con aritmética comprobada.

Un descriptor no copia por sí mismo el contenido dentro del command buffer.
Describe memoria del proceso que el kernel pone a disposición de la operación.
En la implementación actual de servicios internos, Nixe valida el descriptor y
lee o escribe directamente la memoria virtual invitada mediante la interfaz de
memoria emulada.

Ejemplo simplificado de lectura de fichero:

```text
raw CMIF data:
    option = 0
    offset = 0x200
    size   = 0x1000

HIPC receive-buffer descriptor:
    address = 0x80012000
    size    = 0x1000

servicio IFile::Read
    ├── lee como máximo min(petición, descriptor, límite de Nixe)
    ├── escribe bytes en guest[0x80012000..]
    └── devuelve en CMIF el número de bytes leídos
```

## CMIF: comandos y objetos sobre HIPC

CMIF ocupa la zona alineada a 16 bytes de los raw data words de HIPC. Para una
sesión no convertida a dominio, la petición contiene una cabecera de entrada de
16 bytes seguida por argumentos:

```text
CMIF request
┌───────────────────────────────────┐
│ magic = "SFCI"                    │
│ version                           │
│ command ID                        │
│ token                             │
├───────────────────────────────────┤
│ argumentos específicos            │
└───────────────────────────────────┘

CMIF response
┌───────────────────────────────────┐
│ magic = "SFCO"                    │
│ version                           │
│ Horizon result                    │
│ token/contexto usado por Nixe     │
├───────────────────────────────────┤
│ valores de salida                 │
└───────────────────────────────────┘
```

Los magics aparecen como `0x49434653` y `0x4f434653` al leerlos como enteros
little-endian. El token permite correlacionar la respuesta y las variantes
`RequestWithContext`/`ControlWithContext` lo usan como contexto. Nixe valida la
combinación entre versión y tipo de comando.

Nixe escribe actualmente el token de la petición en la última palabra de la
cabecera de salida. En el formato documentado para Horizon 14.0.0 o posterior,
esa posición puede representar un `InterfaceId`. La implementación no
selecciona todavía este detalle según la versión de firmware y debe
considerarse ligada al perfil antiguo que emula.

### Tipos CMIF que reconoce Nixe

| Valor HIPC | Tipo CMIF |
| ---: | --- |
| 1 | Legacy request |
| 2 | Close |
| 3 | Legacy control |
| 4 | Request |
| 5 | Control |
| 6 | Request with context |
| 7 | Control with context |

Los **request** invocan comandos del servicio. Los **control** actúan sobre la
sesión CMIF, no sobre la interfaz concreta; el control 0 convierte una sesión a
dominio. El control 3 consulta el tamaño preferido de pointer buffer. El tipo
close finaliza la sesión.

### Dos clases de resultado

Una llamada tiene un resultado del kernel y, si el transporte llegó al
servicio, otro resultado CMIF:

```text
svcSendSyncRequest
    │
    ├── X0: KernelResult
    │       ¿era válido el handle y se pudo transportar la petición?
    │
    └── CMIF OutHeader.result
            ¿aceptó el servicio el comando y pudo ejecutarlo?
```

Por ejemplo, un handle inexistente produce un error del kernel. En cambio, un
command ID desconocido en una sesión válida puede completar la SVC con éxito y
devolver `CMIF_UNKNOWN_COMMAND_ID` dentro de la respuesta. Mezclar ambos niveles
haría que una aplicación interpretara incorrectamente el fallo.

Los resultados Horizon se codifican con un módulo de 9 bits y una descripción
de 13 bits:

```text
raw result = module | (description << 9)
```

Nixe mantiene códigos semánticos internos independientes de los valores
visibles al invitado y los traduce en la frontera CMIF. Así, un path ausente se
convierte en un resultado del módulo `fs`, mientras que un add-on ausente puede
usar el resultado verificado del módulo `lr`. Los errores de framing u objetos
CMIF usan el módulo `sf`, y los del Service Manager, `sm`.

## Objetos CMIF y dominios

### Sesiones normales

Sin dominio, un objeto hijo suele necesitar un nuevo handle:

```text
handle fsp-srv
    │ OpenDataFileSystemByCurrentProcess
    ▼
handle IFileSystem
    │ OpenFile("/data.bin")
    ▼
handle IFile
```

Cada objeto vivo consume una entrada de la tabla de handles.

### Conversión a dominio

El control CMIF 0 convierte la sesión. A partir de entonces, una sola sesión
transporta mensajes dirigidos a IDs de objeto:

```text
un único handle de sesión
        │
        ├── object ID 1: objeto raíz
        ├── object ID 2: IFileSystem
        ├── object ID 3: IFile
        └── object ID 4: IDirectory
```

Una petición de dominio añade antes de la cabecera CMIF:

- tipo de operación de dominio: enviar mensaje o cerrar objeto;
- cantidad de objetos de entrada;
- tamaño del payload CMIF;
- ID del objeto destino;
- token y campos reservados;
- lista de IDs de objetos de entrada después del payload.

Una respuesta de dominio puede devolver IDs de objetos hijos en vez de nuevos
handles. Esto reduce presión sobre la tabla de handles y conserva la identidad
de todos los objetos bajo la sesión.

Cerrar un objeto de dominio elimina ese ID, no el handle de toda la sesión. El
objeto raíz usa el ID 1 y Nixe no permite cerrarlo como si fuera un hijo.

Nixe limita actualmente una tabla de dominio a 64 objetos. `IpcSession`
mantiene objetos genéricos type-erased, mientras que `AppletSession` mantiene
una tabla especializada de tipos de objeto de applet. Esta duplicidad es una de
las áreas cuya forma podría cambiar al generalizar la implementación.

## Recorrido completo de una llamada

La apertura y lectura de un fichero ilustra todas las capas:

```text
1. Guest/libnx
   ConnectToNamedPort("sm:")
             │
2. SVC 0x1f  │  Nixe crea ServiceManagerSession y devuelve un handle
             ▼
3. sm:RegisterClient (CMIF 0, PID)
             │
4. sm:GetService("fsp-srv") (CMIF 1)
             │  NPDM SAC autoriza el nombre del servicio
             ▼
5. handle IFileSystemProxy
             │ CMIF 1 SetCurrentProcess
             │ CMIF 2 OpenDataFileSystemByCurrentProcess
             │  permisos FS del NPDM autorizan leer content data
             ▼
6. handle/object ID de IFileSystem
             │ CMIF 8 OpenFile
             │ HIPC send-static/send-buffer contiene "/archivo\0"
             ▼
7. handle/object ID de IFile
             │ CMIF 0 Read(offset, size)
             │ HIPC receive-buffer señala el destino
             ▼
8. Nixe lee el ReadOnlyMount/RomFS
             │ escribe en memoria invitada
             ▼
9. respuesta HIPC/CMIF
   KernelResult=Success, CMIF result=Success, bytes_read=N
```

Esta ruta conecta el loader con IPC:

```text
NSP/XCI/NCA procesados por los loaders
                 │
                 ▼
LaunchPlan + política NPDM efectiva
                 │
                 ▼
ProcessMountNamespace
    ├── RomFS base/update
    ├── add-ons autorizados
    ├── Service Access Control (SAC)
    └── permisos de filesystem
                 │
                 ▼
servicios fsp-srv y aoc:u
```

El servicio no abre libremente archivos del host. Opera sobre mounts de solo
lectura ya construidos y autorizados para ese proceso.

## Cómo está organizada la implementación de Nixe

### Vista de componentes

```text
crates/cpu
    └── ejecuta SVC y expone registros/estado del hilo

crates/runtime
    ├── RunnableProcess y memoria virtual
    ├── TLS del hilo
    ├── HandleTable / HandleObject
    ├── PortObject / SessionObject / EventObject / SharedMemoryObject
    └── ProcessMountNamespace y política efectiva

crates/horizon
    ├── svc.rs             registro de números SVC
    ├── svc_dispatch.rs    ABI de kernel, espera, sesiones y transferencia
    ├── ipc_message.rs     codec comprobado HIPC/CMIF
    ├── ipc_wire.rs        puente entre wire, memoria y servicios
    ├── ipc.rs             peticiones/respuestas semánticas
    ├── ipc_result.rs      resultados Horizon visibles al guest
    └── object.rs          objetos y estado de servicios
```

La separación entre `ipc_wire.rs` e `ipc.rs` es deliberada:

```text
bytes no confiables del guest
        │
        ▼
HipcRequest + CmifRequest comprobados
        │
        ▼
IpcRequest tipada y acotada
        │
        ▼
IpcDispatcher semántico
        │
        ▼
IpcResponse tipada
        │
        ▼
CmifResponse comprobada → bytes para el guest
```

Esto permite probar la lógica de filesystem o add-on sin fabricar mensajes
binarios, y probar el codec sin montar contenido real.

### 1. Entrada por SVC

`HorizonSvcDispatcher` reconoce las operaciones IPC principales:

| SVC | Operación implementada |
| ---: | --- |
| `0x1f` | `ConnectToNamedPort` |
| `0x20` | `SendSyncRequestLight` |
| `0x21` | `SendSyncRequest` |
| `0x22` | `SendSyncRequestWithUserBuffer` |
| `0x40` / `0x41` | `CreateSession` / `AcceptSession` |
| `0x42` / `0x43` / `0x44` | variantes de `ReplyAndReceive` |
| `0x70` / `0x71` / `0x72` | crear, administrar y conectar a puertos |
| `0x16` | `CloseHandle` |

Para servicios internos, el dispatcher reconoce el tipo concreto almacenado en
el handle y ejecuta el puente HIPC/CMIF de `ipc_wire.rs`. Para una
`SessionObject` genérica, entrega el mensaje a su endpoint servidor. De este
modo conviven dos caminos:

```text
SendSyncRequest
    │
    ├── handle de servicio interno
    │       └── despacho host-side inmediato y respuesta CMIF
    │
    └── handle de SessionObject genérico
            ├── captura objetos de los handles enviados
            ├── encola la petición
            ├── suspende al cliente
            ├── ReplyAndReceive del servidor invitado
            └── materializa handles de respuesta en el cliente
```

El camino genérico modela servidores invitados y aplica semántica de copia y
movimiento de objetos entre tablas de handles. Las peticiones de cliente no
pueden mover handles; las respuestas de servidor sí pueden copiar y mover,
siguiendo la semántica pública del kernel.

### 2. Decodificación defensiva

`ipc_message.rs`, cuando procesa el command buffer fijo de un servicio
incorporado:

- rechaza mensajes mayores que el command buffer esperado;
- comprueba cada suma, multiplicación y alineación;
- limita descriptores y handles;
- comprueba bits reservados;
- valida que tablas y payloads estén dentro del búfer;
- valida magic, versión, command type y framing de dominio;
- impide adjuntar objetos de dominio a una respuesta no-domain; y
- impide que una respuesta exceda el límite HIPC o el TLS.

Un mensaje mal formado no llega a la capa semántica. Los fallos de acceso a
memoria invitada, framing inválido y agotamiento de recursos se mantienen como
categorías distintas en `IpcWireError`.

### 3. Registro y autorización

Nixe expone `sm:` como puerto nombrado incorporado. Después de
`RegisterClient`, `GetService` puede crear sesiones para:

| Servicio | Cobertura actual resumida |
| --- | --- |
| `fsp-srv` | Apertura del RomFS primario, ficheros y directorios de solo lectura. |
| `aoc:u` | Conteo/listado/preparación y evento de cambio de add-ons autorizados. |
| `set:sys` | Versión de firmware emulada para los comandos usados por libnx. |
| `apm` | Sesión de configuración y modo de rendimiento normal. |
| `appletOE` | Subconjunto del grafo de objetos de aplicación, mediante dominio. |
| `hid` | Creación de `IAppletResource` y memoria compartida HID de solo lectura. |

El acceso se decide en dos pasos:

1. La Service Access Control de la NPDM efectiva debe permitir conectarse al
   servicio.
2. Las operaciones de contenido comprueban además permisos de filesystem como
   `ApplicationInfo`, `ContentManager` o `FullPermission`.

Los homebrew sin NPDM se tratan actualmente como autorizados para acceder al
registro de servicios de la plataforma. Esta política también puede
evolucionar.

### 4. Objetos semánticos

La tabla genérica `HandleTable` almacena objetos type-erased con identidad
compartida. Los servicios pueden devolver:

- `IpcSession`;
- `ReadOnlyFileSystem`;
- `ReadOnlyFile`;
- `ReadOnlyDirectory`;
- eventos;
- memoria compartida;
- sesiones especializadas de settings, rendimiento, applet o HID.

Fuera de un dominio, un hijo se inserta en la tabla y se devuelve su handle.
Dentro de un dominio, Nixe retira el handle temporal de la tabla, conserva el
objeto en la tabla del dominio y devuelve un object ID.

### 5. Filesystem y add-on content

Las operaciones semánticas se expresan con `IpcRequest`/`IpcResponse`. Entre
sus límites actuales están:

| Límite | Valor |
| --- | ---: |
| Path semántico | `0x300` bytes |
| Lectura individual | 1 MiB |
| Entradas por listado | 1024 |
| Entrada de directorio wire | `0x310` bytes |
| Objetos por dominio | 64 |

Los paths deben ser UTF-8, absolutos, no vacíos y canónicos: se rechazan NUL,
componentes vacíos, `.` y `..`, slash final y exceso de longitud. No hay
resolución de paths del host.

Los directorios conservan un cursor compartido. Cada `ReadDirectory` avanza el
cursor, igual que un objeto directorio con estado. Las lecturas de fichero se
recortan al final del archivo y al tamaño real del descriptor de salida.

### 6. Codificación de la respuesta

La capa wire transforma cada `IpcResponse`:

| Respuesta semántica | Representación |
| --- | --- |
| `None` | CMIF success sin payload. |
| `Size` | Entero little-endian en el payload CMIF. |
| `Handle` | Move handle, o object ID si la sesión es un dominio. |
| `Event` | Copy handle. |
| `Data` | Bytes en el receive buffer y cantidad en CMIF. |
| `DirectoryEntries` | Registros de `0x310` bytes en el receive buffer. |
| `AddOnContentEntries` | Índices `u32` en el receive buffer y cantidad en CMIF. |

Si escribir la respuesta en memoria falla después de crear un handle, Nixe
cierra el handle recién creado para no filtrar recursos.

## Servicios y comandos implementados

Esta tabla es un mapa práctico, no una promesa de cobertura estable:

| Objeto | Command IDs principales |
| --- | --- |
| `sm:` | 0 `RegisterClient`, 1 `GetService` |
| `IFileSystemProxy` (`fsp-srv`) | 1 `SetCurrentProcess`, 2 `OpenDataFileSystemByCurrentProcess` |
| `IFileSystem` | 8 `OpenFile`, 9 `OpenDirectory` |
| `IFile` | 0 `Read`, 4 `GetSize` |
| `IDirectory` | 0 `Read`, 1 `GetEntryCount` |
| `aoc:u` | 0/2 count, 1/3 list, 6/7 prepare, 8 changed event |
| `set:sys` | 3/4 firmware version |
| `apm` raíz | 0 open session, 1 get performance mode |
| sesión `apm` | 0 set configuration, 1 get configuration |
| `hid` | 0 create applet resource |
| `IAppletResource` | 0 get shared memory handle |

`appletOE` implementa la conversión a dominio, la apertura de
`IApplicationProxy` y un subconjunto de `ICommonStateGetter`,
`ISelfController`, `IWindowController` e `IApplicationFunctions`. Otros hijos
pueden existir en la tabla de dominio pero responder todavía
`CMIF_UNKNOWN_COMMAND_ID`.

## Qué no debe asumirse todavía

La cobertura actual no equivale a una emulación completa de Horizon IPC:

- no todos los servicios ni comandos de Switch están implementados;
- TIPC no está implementado como protocolo general;
- `SendSyncRequestWithUserBuffer` funciona como transporte de sesiones
  genéricas, pero el codec fijo de servicios incorporados no acepta todavía su
  región de página completa;
- el codec asume que la receive-static list sigue a los raw data y no aplica
  todavía todas las variantes de `ReceiveListOffset`;
- el puente de servicios internos no reproduce todavía todas las reglas de
  mapeo y cacheabilidad que aplicaría el kernel real a cada descriptor;
- muchos comandos solo aceptan las formas de descriptor usadas por libnx en
  las rutas probadas;
- la emulación de applet, HID y rendimiento devuelve un conjunto mínimo de
  datos coherentes, no el comportamiento completo del hardware;
- no hay selección general de ABI de servicio según versión de firmware;
- algunos límites son defensivos de Nixe y no límites normativos de Horizon;
- las tablas especializadas y genéricas de dominio podrían unificarse; y
- los nombres y tipos públicos de la capa semántica pueden cambiar al ampliar
  servicios de escritura, storage, GPU, audio o red.

Por estas razones, al modificar la implementación deben revisarse al menos las
tablas de comandos, los diagramas de despacho, los límites y la sección de
resultados de este documento.

## Cómo orientarse al depurar

Ante un fallo IPC conviene identificar primero la capa:

| Síntoma | Capa probable |
| --- | --- |
| La SVC devuelve `InvalidHandle` | Tabla de handles, endpoint o tipo de sesión. |
| El hilo queda suspendido indefinidamente | Cola de sesión, servidor, wait/reply o cierre del peer. |
| La SVC tiene éxito pero CMIF devuelve error | Command ID, argumentos, permisos u operación del servicio. |
| Se rechaza el mensaje antes del servicio | Cabecera HIPC, offsets, magic/version CMIF o dominio. |
| El servicio devuelve éxito pero el buffer no cambia | Dirección/tamaño/modo del descriptor o memoria invitada. |
| Un objeto hijo desaparece o agota handles | Semántica copy/move, cierre o conversión a dominio. |
| `sm:GetService` rechaza el nombre | Registro del servicio o SAC de NPDM. |
| `fsp-srv` conecta pero no abre contenido | Permiso FS de NPDM o mount ausente. |

El log de `ipc_wire.rs` incluye handle, tipo, command ID, presencia de PID y
conteos de descriptores y handles. Esa información permite distinguir
rápidamente framing, routing y semántica.

## Referencias

- [Switchbrew: HIPC](https://switchbrew.org/wiki/HIPC), descripción de HIPC,
  CMIF, cabeceras, descriptores y dominios.
- [Switchbrew: SVC](https://switchbrew.org/wiki/SVC), ABI pública de las
  supervisor calls de Horizon.
- [libnx: `sf/service.h`](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/service.h),
  construcción y ciclo de vida de servicios CMIF usado como referencia fijada
  por la implementación.
- [libnx: `sf/cmif.h`](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/cmif.h),
  estructuras y helpers CMIF.
- [libnx: `sf/hipc.h`](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/hipc.h),
  estructuras y helpers HIPC.
- [Atmosphère: IPC del kernel](https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/svc/kern_svc_ipc.cpp),
  referencia pública para sesiones, puertos y SVC IPC.
- [Atmosphère: resultados comunes](https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libvapours/include/vapours/results/results_common.hpp),
  codificación de resultados Horizon.
