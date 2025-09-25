# NNPipeline Architecture

## 개요
동적 네트워크 구성을 위한 기반 패키지로, Fluent Builder Pattern과 Factory Pattern을 조합하여 설정과 구현을 분리한 구조

## 전체 시스템 구조

```mermaid
graph TB
    User[사용자] --> PipeBuilder[PipeBuilder]
    PipeBuilder --> ConfigString[Config String]
    ConfigString --> PipeFactory[PipeFactory]
    PipeFactory --> SubFactory1[LinearFactory]
    PipeFactory --> SubFactory2[NormalizationFactory]
    PipeFactory --> SubFactory3[ActivationFactory]
    PipeFactory --> SubFactory4[DropoutFactory]
    PipeFactory --> SubFactory5[ExponentialFactory]

    SubFactory1 --> LinearLayer[nn.Linear]
    SubFactory2 --> BatchNorm[nn.BatchNorm1d]
    SubFactory2 --> LayerNorm[nn.LayerNorm]
    SubFactory3 --> ReLU[nn.ReLU]
    SubFactory3 --> GELU[nn.GELU]
    SubFactory4 --> Dropout[nn.Dropout]
    SubFactory5 --> ExpEncoder[LinearExponentialEncoder]
    SubFactory5 --> ExpDecoder[LinearExponentialDecoder]

    PipeFactory --> FinalPipe[완성된 Pipe]
```

## 핵심 컴포넌트

### 1. PipeBuilder (Fluent Interface)

```mermaid
classDiagram
    class PipeBuilder {
        -config_parts: List[str]
        +linear(input: int, output: int) PipeBuilder
        +batch_norm() PipeBuilder
        +layer_norm() PipeBuilder
        +relu() PipeBuilder
        +gelu() PipeBuilder
        +dropout(rate: float) PipeBuilder
        +exponential_encoder(input: int, output: int, rate: float) PipeBuilder
        +exponential_decoder(input: int, output: int, rate: float) PipeBuilder
        +serialize() str
    }
```

### 2. Factory 계층 구조

```mermaid
classDiagram
    class PipeFactory {
        -sub_factories: Dict[str, SubFactory]
        +register_factory(name: str, factory: SubFactory)
        +create(config_string: str) Pipe
        -parse_config(config_string: str) List[LayerConfig]
    }

    class SubFactory {
        <<interface>>
        +can_handle(layer_type: str) bool
        +create_layer(config: LayerConfig) nn.Module
    }

    class LinearFactory {
        +can_handle(layer_type: str) bool
        +create_layer(config: LayerConfig) nn.Linear
        -parse_linear_config(config: str) dict
    }

    class NormalizationFactory {
        +can_handle(layer_type: str) bool
        +create_layer(config: LayerConfig) nn.Module
        -create_batch_norm(features: int) nn.BatchNorm1d
        -create_layer_norm(features: int) nn.LayerNorm
    }

    class ActivationFactory {
        +can_handle(layer_type: str) bool
        +create_layer(config: LayerConfig) nn.Module
    }

    class DropoutFactory {
        +can_handle(layer_type: str) bool
        +create_layer(config: LayerConfig) nn.Dropout
    }

    class ExponentialFactory {
        +can_handle(layer_type: str) bool
        +create_layer(config: LayerConfig) nn.Module
        -create_encoder(config: dict) LinearExponentialEncoder
        -create_decoder(config: dict) LinearExponentialDecoder
    }

    PipeFactory --> SubFactory
    SubFactory <|-- LinearFactory
    SubFactory <|-- NormalizationFactory
    SubFactory <|-- ActivationFactory
    SubFactory <|-- DropoutFactory
    SubFactory <|-- ExponentialFactory
```

### 3. 설정 문자열 파싱 구조

```mermaid
flowchart TD
    ConfigString["linear(784->256)|batch_norm|relu|dropout(0.3)"]
    ConfigString --> Parser[ConfigParser]
    Parser --> LayerConfig1[LayerConfig: linear]
    Parser --> LayerConfig2[LayerConfig: batch_norm]
    Parser --> LayerConfig3[LayerConfig: relu]
    Parser --> LayerConfig4[LayerConfig: dropout]

    LayerConfig1 --> LinearFactory
    LayerConfig2 --> NormalizationFactory
    LayerConfig3 --> ActivationFactory
    LayerConfig4 --> DropoutFactory

    LinearFactory --> nn.Linear
    NormalizationFactory --> nn.BatchNorm1d
    ActivationFactory --> nn.ReLU
    DropoutFactory --> nn.Dropout
```

## 데이터 흐름

```mermaid
sequenceDiagram
    participant User
    participant Builder as PipeBuilder
    participant Factory as PipeFactory
    participant SubFactory as LinearFactory
    participant Layer as nn.Linear

    User->>Builder: .linear(784, 256)
    User->>Builder: .batch_norm()
    User->>Builder: .serialize()
    Builder->>User: "linear(784->256)|batch_norm"

    User->>Factory: .create(config_string)
    Factory->>Factory: parse_config()
    Factory->>SubFactory: create_layer(linear_config)
    SubFactory->>Layer: nn.Linear(784, 256)
    Layer->>SubFactory: linear_layer
    SubFactory->>Factory: linear_layer
    Factory->>User: completed_pipe
```

## 확장성 설계

### 새로운 레이어 타입 추가 방법
1. 새로운 SubFactory 구현
2. PipeBuilder에 해당 메소드 추가
3. PipeFactory에 SubFactory 등록
4. 파싱 규칙 정의

### 설정 문자열 형식
- 기본 형태: `layer_type(param1->param2, param3=value)`
- 구분자: `|` (파이프)
- 파라미터: `()` 내부에 정의
- 예시: `linear(784->256)|batch_norm|relu|dropout(0.3)`
