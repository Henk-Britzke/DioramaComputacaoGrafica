#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <assert.h>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

// GLAD
#include <glad/glad.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// GLFW
#include <GLFW/glfw3.h>

//GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Protótipo da função de callback de teclado
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);

struct MaterialInfo {
    glm::vec3 Ka{0.2f};
    glm::vec3 Kd{1.0f};
    glm::vec3 Ks{0.5f};
    float Ns = 32.0f;
    string diffuseTexture;
    string opacityTexture;
};

struct Mesh {
    GLuint VAO = 0;
    GLuint VBO = 0;
    int vertexCount = 0;
    GLuint textureID = 0;
    bool textured = false;
    MaterialInfo material;
    glm::vec3 baseColor{1.0f};
};

// Protótipos das funções
int setupShader();
struct ObjModel;
bool loadSimpleOBJ(const string& filePath, vector<Mesh>& outMeshes, const glm::vec3& defaultColor);
void updateThreePointLights(const ObjModel& mainObject);
bool loadTrajectories(const string& filePath);
bool saveTrajectories(const string& filePath);
void updateObjectTrajectory(ObjModel& obj, float deltaTime);
glm::vec3 bezierCubic(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3, float t);
bool loadSceneConfig(const string& filePath);
fs::path findProjectRoot(const fs::path& startDir, const string& markerFile);

// Dimensões da janela (pode ser alterado em tempo de execução)
const GLuint WIDTH = 1000, HEIGHT = 1000;
const string TRAJECTORY_FILE = "trajectories.txt";

class Camera {
public:
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 worldUp;
    float yaw;
    float pitch;
    float speed;
    float sensitivity;

    Camera(glm::vec3 startPosition = glm::vec3(0.0f, 0.0f, 5.0f))
        : Camera(startPosition, -90.0f, 0.0f)
    {
    }

    Camera(glm::vec3 startPosition, float startYaw, float startPitch)
        : position(startPosition),
          front(glm::vec3(0.0f, 0.0f, -1.0f)),
          worldUp(glm::vec3(0.0f, 1.0f, 0.0f)),
          yaw(startYaw),
          pitch(startPitch),
          speed(3.0f),
          sensitivity(0.1f)
    {
        updateVectors();
    }

    glm::mat4 getViewMatrix() const
    {
        return glm::lookAt(position, position + front, up);
    }

    void Mover(GLFWwindow* window, float deltaTime)
    {
        float velocity = speed * deltaTime;

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            position += front * velocity;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            position -= front * velocity;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            position -= right * velocity;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            position += right * velocity;
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
            position += worldUp * velocity;
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
            position -= worldUp * velocity;
    }

    void Rotacionar(float xoffset, float yoffset)
    {
        yaw += xoffset * sensitivity;
        pitch += yoffset * sensitivity;

        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;

        updateVectors();
    }

    void setOrientation(float newYaw, float newPitch)
    {
        yaw = newYaw;
        pitch = newPitch;

        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;

        updateVectors();
    }

private:
    void updateVectors()
    {
        glm::vec3 direction;
        direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        direction.y = sin(glm::radians(pitch));
        direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));

        front = glm::normalize(direction);
        right = glm::normalize(glm::cross(front, worldUp));
        up = glm::normalize(glm::cross(right, front));
    }
};

// Código fonte do Vertex Shader (em GLSL)
const GLchar* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 position;\n"
"layout (location = 1) in vec3 color;\n"
"layout (location = 2) in vec2 texCoord;\n"
"layout (location = 3) in vec3 normal;\n"
"uniform mat4 model;\n"
"uniform mat4 view;\n"
"uniform mat4 projection;\n"
"uniform bool useOverrideColor;\n"
"uniform vec4 overrideColor;\n"
"out vec4 finalColor;\n"
"out vec2 TexCoord;\n"
"out vec3 FragPos;\n"
"out vec3 Normal;\n"
"void main()\n"
"{\n"
"    vec4 worldPos = model * vec4(position, 1.0);\n"
"    gl_Position = projection * view * worldPos;\n"
"    FragPos = vec3(worldPos);\n"
"    Normal = mat3(transpose(inverse(model))) * normal;\n"
"    finalColor = useOverrideColor ? overrideColor : vec4(color, 1.0);\n"
"    TexCoord = texCoord;\n"
"}\0";

//Código fonte do Fragment Shader (em GLSL)
const GLchar* fragmentShaderSource = "#version 330 core\n"
"in vec4 finalColor;\n"
"in vec2 TexCoord;\n"
"in vec3 FragPos;\n"
"in vec3 Normal;\n"
"uniform sampler2D texBuff;\n"
"uniform bool useTexture;\n"
"const int NUM_LIGHTS = 3;\n"
"uniform vec3 lightPos[NUM_LIGHTS];\n"
"uniform vec3 lightColor[NUM_LIGHTS];\n"
"uniform float lightIntensity[NUM_LIGHTS];\n"
"uniform int lightEnabled[NUM_LIGHTS];\n"
"uniform vec3 viewPos;\n"
"uniform vec3 Ka;\n"
"uniform vec3 Kd;\n"
"uniform vec3 Ks;\n"
"uniform float Ns;\n"
"uniform float Kc;\n"
"uniform float Kl;\n"
"uniform float Kq;\n"
"out vec4 color;\n"
"void main()\n"
"{\n"
"    vec3 objectColor = finalColor.rgb;\n"
"    if (useTexture) {\n"
"        objectColor = texture(texBuff, TexCoord).rgb;\n"
"    }\n"
"    vec3 N = normalize(Normal);\n"
"    vec3 V = normalize(viewPos - FragPos);\n"
"    vec3 result = vec3(0.0);\n"
"    for (int i = 0; i < NUM_LIGHTS; i++) {\n"
"        if (lightEnabled[i] == 0) {\n"
"            continue;\n"
"        }\n"
"        vec3 L = normalize(lightPos[i] - FragPos);\n"
"        vec3 R = reflect(-L, N);\n"
"        float d = length(lightPos[i] - FragPos);\n"
"        float attenuation = 1.0 / (Kc + Kl * d + Kq * d * d);\n"
"        vec3 ambient = 0.12 * Ka * lightColor[i] * lightIntensity[i] * objectColor;\n"
"        float diff = max(dot(N, L), 0.0);\n"
"        vec3 diffuse = Kd * diff * lightColor[i] * lightIntensity[i] * attenuation * objectColor;\n"
"        float spec = pow(max(dot(V, R), 0.0), Ns);\n"
"        vec3 specular = Ks * spec * lightColor[i] * lightIntensity[i];\n"
"        result += ambient + diffuse + specular;\n"
"    }\n"
"    color = vec4(result, finalColor.a);\n"
"}\n\0";

struct ObjModel {
    string name;
    vector<Mesh> subMeshes;
    glm::vec3 translation{0.0f};
    glm::vec3 rotation{0.0f};
    glm::vec3 rotationSpeed{0.0f};
    glm::vec3 scale{1.0f};
    glm::vec3 color{1.0f};
    bool selected = false;
    std::vector<glm::vec3> trajectoryPoints;
    int currentTrajectoryIndex = 0;
    float trajectorySpeed = 1.2f;
    bool followTrajectory = false;
    // Curva de Bezier cubico
    bool useBezier = false;  
    float trajectoryT = 0.0f;
    // Controle de rotacao
    bool rotationPaused = false;         // pausa/retoma a rotacao automatica
    glm::vec3 savedRotationSpeed{0.0f};  // guarda a velocidade ao pausar
};

struct PointLight {
    glm::vec3 position{0.0f};
    glm::vec3 color{1.0f};
    float intensity = 1.0f;
    bool enabled = true;
};

const int NUM_LIGHTS = 3;
PointLight sceneLights[NUM_LIGHTS];
vector<ObjModel> sceneObjects;
int selectedIndex = 0;
int mainObjectIndex = 0;
Camera camera;
glm::vec3 configuredCameraPosition(0.0f, 0.0f, 5.0f);
float configuredCameraYaw = -90.0f;
float configuredCameraPitch = 0.0f;
float configuredCameraFov = 45.0f;
float configuredCameraNearPlane = 0.1f;
float configuredCameraFarPlane = 100.0f;
glm::vec3 configuredBackgroundColor(0.95f, 0.95f, 0.95f);

// Coeficientes de iluminacao globais (parametrizaveis por teclado)
float globalKa = 1.0f;
float globalKd = 1.0f;
float globalKs = 1.0f;
float globalNs = 1.0f;

bool firstMouse = true;
double lastMouseX = WIDTH / 2.0;
double lastMouseY = HEIGHT / 2.0;

void printHelp()
{
    cout << "Mouse     : olhar ao redor\n";
    cout << "W/A/S/D   : mover câmera\n";
    cout << "Espaco/Shift : subir/descer câmera\n";
    cout << "Setas/I/J : transladar objeto\n";
    cout << "Z/X/C     : iniciar/pausar/retomar rotacao (eixo Z/X/Y)\n";
    cout << "T         : iniciar/pausar trajetoria\n";
    cout << "[]       : Escala uniforme maior|menor\n";
    cout << "ligar/desligar luz: tecla 1(principal) / tecla 2(preenchimento) / tecla 3 (fundo)\n";
    cout << "TAB             : trocar objeto selecionado\n";
    cout << "P               : adicionar ponto de trajetoria\n";
    cout << "O               : limpar pontos de trajetória\n";
    cout << "L               : salvar trajetórias\n";
    cout << "ESC       : sair\n";
    cout << "--- Iluminacao ---\n";
    cout << "V / Ctrl+V  : Ka (ambiente) maior/menor\n";
    cout << "B / Ctrl+B  : Kd (difusa) maior/menor\n";
    cout << "N / Ctrl+N  : Ks (especular) maior/menor\n";
    cout << "M / Ctrl+M  : Ns (brilho especular) maior/menor\n";
}

ObjModel& selectedObject()
{
    assert(!sceneObjects.empty());
    return sceneObjects[selectedIndex];
}

fs::path findProjectRoot(const fs::path& startDir, const string& markerFile)
{
    std::error_code ec;
    fs::path current = fs::absolute(startDir, ec);
    if (ec) {
        current = startDir;
    }

    while (true) {
        if (fs::exists(current / markerFile)) {
            return current;
        }

        if (!current.has_parent_path() || current.parent_path() == current) {
            break;
        }
        current = current.parent_path();
    }

    return {};
}


int main()
{
    if (!glfwInit()) {
        cerr << "Falha ao inicializar GLFW" << endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Trajetoria", nullptr, nullptr);
    if (!window) {
        cerr << "Falha ao criar janela GLFW" << endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        cerr << "Failed to initialize GLAD" << endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    fs::path projectRoot = findProjectRoot(fs::current_path(), "scene.json");
    if (!projectRoot.empty()) {
        fs::current_path(projectRoot);
    } else {
        cerr << "Aviso: nao foi possivel localizar a raiz do projeto a partir de "
             << fs::current_path() << endl;
    }

    stbi_set_flip_vertically_on_load(true);
    //
    //const GLubyte* renderer = glGetString(GL_RENDERER);
    //const GLubyte* version = glGetString(GL_VERSION);
    //cout << "Renderer: " << renderer << endl;
    //cout << "OpenGL version supported " << version << endl;

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);

    GLuint shaderID = setupShader();
    glUseProgram(shaderID);
    //criacao de variaveis para guardar as localizacoes (IDs) das variaveis uniformes no shader, para que possam ser usadas posteriormente para enviar os dados de transformacao e iluminacao para o shader
    GLint modelLoc = glGetUniformLocation(shaderID, "model");
    GLint viewLoc = glGetUniformLocation(shaderID, "view");
    GLint projLoc = glGetUniformLocation(shaderID, "projection");
    GLint overrideLoc = glGetUniformLocation(shaderID, "useOverrideColor");
    GLint overrideColorLoc = glGetUniformLocation(shaderID, "overrideColor");
    GLint texUniformLoc = glGetUniformLocation(shaderID, "texBuff");
    GLint useTextureLoc = glGetUniformLocation(shaderID, "useTexture");
    GLint viewPosLoc = glGetUniformLocation(shaderID, "viewPos");
    GLint kaLoc = glGetUniformLocation(shaderID, "Ka");
    GLint kdLoc = glGetUniformLocation(shaderID, "Kd");
    GLint ksLoc = glGetUniformLocation(shaderID, "Ks");
    GLint nsLoc = glGetUniformLocation(shaderID, "Ns");
    GLint kcLoc = glGetUniformLocation(shaderID, "Kc");
    GLint klLoc = glGetUniformLocation(shaderID, "Kl");
    GLint kqLoc = glGetUniformLocation(shaderID, "Kq");
    GLint lightPosLoc[NUM_LIGHTS];
    GLint lightColorLoc[NUM_LIGHTS];
    GLint lightIntensityLoc[NUM_LIGHTS];
    GLint lightEnabledLoc[NUM_LIGHTS];
    for (int i = 0; i < NUM_LIGHTS; i++) {
        string index = to_string(i);
        lightPosLoc[i] = glGetUniformLocation(shaderID, ("lightPos[" + index + "]").c_str());
        lightColorLoc[i] = glGetUniformLocation(shaderID, ("lightColor[" + index + "]").c_str());
        lightIntensityLoc[i] = glGetUniformLocation(shaderID, ("lightIntensity[" + index + "]").c_str());
        lightEnabledLoc[i] = glGetUniformLocation(shaderID, ("lightEnabled[" + index + "]").c_str());
    }
    // //enviar para o shader.
    glUniform1i(texUniformLoc, 0);
    glUniform1i(useTextureLoc, 0);
    glUniform1f(kcLoc, 1.0f);
    glUniform1f(klLoc, 0.09f);
    glUniform1f(kqLoc, 0.032f);

    glEnable(GL_DEPTH_TEST);

    printHelp();

    if (!loadSceneConfig("scene.json")) {
        cerr << "Falha ao carregar configuração de cena scene.json" << endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    glm::mat4 projection = glm::perspective(
        glm::radians(configuredCameraFov),
        static_cast<float>(WIDTH) / static_cast<float>(HEIGHT),
        configuredCameraNearPlane,
        configuredCameraFarPlane
    );
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

    if (sceneObjects.empty()) {
        cerr << "Nenhum modelo carregado. Verifique o arquivo de cena ou arquivos OBJ." << endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    loadTrajectories(TRAJECTORY_FILE);

    mainObjectIndex = 0;
    selectedIndex = 0;
    sceneObjects[selectedIndex].selected = true;

    double lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        float deltaTime = static_cast<float>(currentTime - lastTime);
        lastTime = currentTime;

        glfwPollEvents();
        camera.Mover(window, deltaTime);

        for (auto& obj : sceneObjects) {
            if (obj.followTrajectory && !obj.trajectoryPoints.empty()) {
                updateObjectTrajectory(obj, deltaTime);
            }
            if (!obj.rotationPaused) {
                obj.rotation += obj.rotationSpeed * deltaTime;
            }
        }
        updateThreePointLights(sceneObjects[mainObjectIndex]);

        glm::mat4 view = camera.getViewMatrix();
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniform3fv(viewPosLoc, 1, glm::value_ptr(camera.position));

        glClearColor(configuredBackgroundColor.r, configuredBackgroundColor.g, configuredBackgroundColor.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        for (int i = 0; i < NUM_LIGHTS; i++) {
            glUniform3fv(lightPosLoc[i], 1, glm::value_ptr(sceneLights[i].position));
            glUniform3fv(lightColorLoc[i], 1, glm::value_ptr(sceneLights[i].color));
            glUniform1f(lightIntensityLoc[i], sceneLights[i].intensity);
            glUniform1i(lightEnabledLoc[i], sceneLights[i].enabled ? 1 : 0);
        }

        for (auto& obj : sceneObjects) {
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, obj.translation);
            model = glm::rotate(model, obj.rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
            model = glm::rotate(model, obj.rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
            model = glm::rotate(model, obj.rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
            model = glm::scale(model, obj.scale);

            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
            for (auto& submesh : obj.subMeshes) {
                // Coeficientes multiplicados pelos fatores globais (controlados por V/B/N/M)
                glm::vec3 kaFinal = submesh.material.Ka * globalKa;
                glm::vec3 kdFinal = submesh.material.Kd * globalKd;
                glm::vec3 ksFinal = submesh.material.Ks * globalKs;
                float nsFinal    = submesh.material.Ns  * globalNs;
                glUniform3fv(kaLoc, 1, glm::value_ptr(kaFinal));
                glUniform3fv(kdLoc, 1, glm::value_ptr(kdFinal));
                glUniform3fv(ksLoc, 1, glm::value_ptr(ksFinal));
                glUniform1f(nsLoc, nsFinal);

                // Cor do submesh: selecionado = amarelo; sem textura = Kd do material; com textura = textura
                if (obj.selected) {
                    glUniform1i(overrideLoc, 1);
                    glUniform4f(overrideColorLoc, 1.0f, 1.0f, 0.2f, 1.0f);
                } else if (!submesh.textured) {
                    // Usa Kd do material como cor solida do submesh
                    glUniform1i(overrideLoc, 1);
                    glUniform4f(overrideColorLoc,
                        submesh.material.Kd.r,
                        submesh.material.Kd.g,
                        submesh.material.Kd.b,
                        1.0f);
                } else {
                    glUniform1i(overrideLoc, 0);
                }

                glUniform1i(useTextureLoc, submesh.textured ? 1 : 0);
                if (submesh.textured && submesh.textureID != 0) {
                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_2D, submesh.textureID);
                } else {
                    glBindTexture(GL_TEXTURE_2D, 0);
                }

                glBindVertexArray(submesh.VAO);
                glDrawArrays(GL_TRIANGLES, 0, submesh.vertexCount);
                glBindVertexArray(0);

                if (submesh.textured && submesh.textureID != 0) {
                    glBindTexture(GL_TEXTURE_2D, 0);
                }
            }
        }

        glfwSwapBuffers(window);
    }

    saveTrajectories(TRAJECTORY_FILE);
    for (const auto& obj : sceneObjects) {
        for (const auto& submesh : obj.subMeshes) {
            if (submesh.textureID != 0) {
                glDeleteTextures(1, &submesh.textureID);
            }
            if (submesh.VAO != 0) {
                glDeleteVertexArrays(1, &submesh.VAO);
            }
            if (submesh.VBO != 0) {
                glDeleteBuffers(1, &submesh.VBO);
            }
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
//configuracao das posicoes das luzes de 3 pontos com base na posicao e escala do objeto
void updateThreePointLights(const ObjModel& mainObject)
{
    float objectScale = glm::max(mainObject.scale.x, glm::max(mainObject.scale.y, mainObject.scale.z));
    float distance = glm::max(objectScale * 2.5f, 2.0f);
    glm::vec3 center = mainObject.translation;

    sceneLights[0].position = center + glm::vec3(-distance, distance * 0.9f, distance);
    sceneLights[0].color = glm::vec3(1.0f, 0.95f, 0.88f);
    sceneLights[0].intensity = 1.25f;

    sceneLights[1].position = center + glm::vec3(distance, distance * 0.45f, distance * 0.65f);
    sceneLights[1].color = glm::vec3(0.75f, 0.85f, 1.0f);
    sceneLights[1].intensity = 0.45f;

    sceneLights[2].position = center + glm::vec3(0.0f, distance * 0.8f, -distance);
    sceneLights[2].color = glm::vec3(1.0f, 1.0f, 1.0f);
    sceneLights[2].intensity = 0.85f;
}
// acao movimento do mouse
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse) {
        lastMouseX = xpos;
        lastMouseY = ypos;
        firstMouse = false;
    }

    float xoffset = static_cast<float>(xpos - lastMouseX);
    float yoffset = static_cast<float>(lastMouseY - ypos);
    lastMouseX = xpos;
    lastMouseY = ypos;

    camera.Rotacionar(xoffset, yoffset);
}
//acoes do teclado
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT) {
        return;
    }

    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(window, GL_TRUE);
        return;
    }

    if (action == GLFW_PRESS && key >= GLFW_KEY_1 && key <= GLFW_KEY_3) {
        int lightIndex = key - GLFW_KEY_1;
        sceneLights[lightIndex].enabled = !sceneLights[lightIndex].enabled;
    }

    if (sceneObjects.empty()) {
        return;
    }

    if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
        sceneObjects[selectedIndex].selected = false;
        selectedIndex = (selectedIndex + 1) % static_cast<int>(sceneObjects.size());
        sceneObjects[selectedIndex].selected = true;
        cout << "Objeto selecionado: " << sceneObjects[selectedIndex].name << endl;
        return;
    }

    ObjModel& obj = selectedObject();

    if (key == GLFW_KEY_LEFT) {
        obj.translation.x -= 0.1f;
    }
    if (key == GLFW_KEY_RIGHT) {
        obj.translation.x += 0.1f;
    }
    if (key == GLFW_KEY_UP) {
        obj.translation.z -= 0.1f;
    }
    if (key == GLFW_KEY_DOWN) {
        obj.translation.z += 0.1f;
    }
    if (key == GLFW_KEY_I) {
        obj.translation.y += 0.1f;
    }
    if (key == GLFW_KEY_J) {
        obj.translation.y -= 0.1f;
    }

    if (key == GLFW_KEY_P && action == GLFW_PRESS) {
        obj.trajectoryPoints.push_back(obj.translation);
        cout << "Ponto de trajetoria adicionado para " << obj.name << ": ("
             << obj.translation.x << ", " << obj.translation.y << ", " << obj.translation.z << ")\n";
    }
    if (key == GLFW_KEY_T && action == GLFW_PRESS) {
        if (obj.trajectoryPoints.empty()) {
            cout << "Nenhum ponto de trajetoria para " << obj.name << ". Use P para adicionar pontos." << endl;
        } else {
            obj.followTrajectory = !obj.followTrajectory;
            // Ao retomar, reseta o parametro t para continuar suavemente do ponto atual
            if (obj.followTrajectory && obj.useBezier) {
                // nao reseta trajectoryT para continuar de onde parou
            }
            cout << "Trajetoria " << (obj.followTrajectory ? "iniciada" : "pausada") << " para " << obj.name << endl;
        }
    }
    if (key == GLFW_KEY_O && action == GLFW_PRESS) {
        obj.trajectoryPoints.clear();
        obj.currentTrajectoryIndex = 0;
        obj.followTrajectory = false;
        cout << "Pontos de trajetoria removidos de " << obj.name << endl;
    }
    if (key == GLFW_KEY_L && action == GLFW_PRESS) {
        if (saveTrajectories(TRAJECTORY_FILE)) {
            cout << "Trajetorias salvas em " << TRAJECTORY_FILE << endl;
        } else {
            cout << "Falha ao salvar trajetorias em " << TRAJECTORY_FILE << endl;
        }
    }

    // Z/X/C: toggle de rotacao no eixo correspondente
    // - Se o objeto nao esta girando nesse eixo: inicia
    // - Se ja esta girando nesse eixo: pausa/retoma
    auto toggleRotation = [&](int axis) {
        glm::vec3 speed(0.0f);
        speed[axis] = glm::radians(90.0f);
        bool mesmoEixo = (glm::length(obj.savedRotationSpeed - speed) < 0.001f ||
                          glm::length(obj.rotationSpeed - speed) < 0.001f);
        if (mesmoEixo && (obj.rotationPaused || glm::length(obj.rotationSpeed) > 0.0f)) {
            // Toggle pausa/retoma no mesmo eixo
            if (obj.rotationPaused) {
                obj.rotationSpeed = obj.savedRotationSpeed;
                obj.rotationPaused = false;
                cout << "Rotacao retomada para " << obj.name << endl;
            } else {
                obj.savedRotationSpeed = obj.rotationSpeed;
                obj.rotationSpeed = glm::vec3(0.0f);
                obj.rotationPaused = true;
                cout << "Rotacao pausada para " << obj.name << endl;
            }
        } else {
            // Inicia rotacao no novo eixo
            obj.rotation = glm::vec3(0.0f);
            obj.rotationSpeed = speed;
            obj.savedRotationSpeed = speed;
            obj.rotationPaused = false;
            cout << "Rotacao iniciada (eixo " << axis << ") para " << obj.name << endl;
        }
    };
    if (key == GLFW_KEY_Z && action == GLFW_PRESS) toggleRotation(2);
    if (key == GLFW_KEY_X && action == GLFW_PRESS) toggleRotation(0);
    if (key == GLFW_KEY_C && action == GLFW_PRESS) toggleRotation(1);

    if (key == GLFW_KEY_LEFT_BRACKET) {
        obj.scale *= 1.1f;
    }
    if (key == GLFW_KEY_RIGHT_BRACKET) {
        obj.scale *= 0.9f;
    }

    // --- Coeficientes de iluminacao (V/B/N/M aumenta, Ctrl+V/B/N/M diminui) ---
    const float STEP_K  = 0.05f;  // passo para Ka, Kd, Ks
    const float STEP_NS = 0.1f;   // passo para Ns (multiplicador)
    bool ctrlHeld = (mode & GLFW_MOD_CONTROL) != 0;

    if (key == GLFW_KEY_V) {
        globalKa = glm::clamp(globalKa + (ctrlHeld ? -STEP_K : STEP_K), 0.0f, 2.0f);
        cout << "Ka: " << globalKa << endl;
    }
    if (key == GLFW_KEY_B) {
        globalKd = glm::clamp(globalKd + (ctrlHeld ? -STEP_K : STEP_K), 0.0f, 2.0f);
        cout << "Kd: " << globalKd << endl;
    }
    if (key == GLFW_KEY_N) {
        globalKs = glm::clamp(globalKs + (ctrlHeld ? -STEP_K : STEP_K), 0.0f, 2.0f);
        cout << "Ks: " << globalKs << endl;
    }
    if (key == GLFW_KEY_M) {
        globalNs = glm::clamp(globalNs + (ctrlHeld ? -STEP_NS : STEP_NS), 0.1f, 10.0f);
        cout << "Ns (multiplicador): " << globalNs << endl;
    }
}

int setupShader()
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);


    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

bool loadTrajectories(const string& filePath)
{
    ifstream file(filePath);
    if (!file.is_open()) {
        return false;
    }

    string line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        istringstream ss(line);
        string name;
        float x, y, z;
        if (!(ss >> name >> x >> y >> z)) continue;

        for (auto& obj : sceneObjects) {
            if (obj.name == name) {
                obj.trajectoryPoints.emplace_back(x, y, z);
                break;
            }
        }
    }
    file.close();
    return true;
}

bool saveTrajectories(const string& filePath)
{
    ofstream file(filePath);
    if (!file.is_open()) {
        return false;
    }
    for (const auto& obj : sceneObjects) {
        for (const auto& point : obj.trajectoryPoints) {
            file << obj.name << " " << point.x << " " << point.y << " " << point.z << "\n";
        }
    }
    file.close();
    return true;
}

// Avalia um ponto na curva de Bezier cubica dado 4 pontos de controle e t em [0,1]
glm::vec3 bezierCubic(const glm::vec3& p0, const glm::vec3& p1,
                      const glm::vec3& p2, const glm::vec3& p3, float t)
{
    float u = 1.0f - t;
    return u*u*u*p0
         + 3.0f*u*u*t*p1
         + 3.0f*u*t*t*p2
         + t*t*t*p3;
}

void updateObjectTrajectory(ObjModel& obj, float deltaTime)
{
    if (obj.trajectoryPoints.empty()) {
        return;
    }

    // Bezier
    // requer multiplos de 4 pontos de controle no array trajectory do JSON.
    // cada grupo de 4 pontos define um segmento: P0(ancora), P1(controle), P2(controle), P3(ancora).
    if (obj.useBezier && obj.trajectoryPoints.size() >= 4) {
        int numSegments = static_cast<int>(obj.trajectoryPoints.size()) / 4;

        // Avanca t pelo tempo
        obj.trajectoryT += obj.trajectorySpeed * deltaTime * 0.25f;

        if (obj.trajectoryT >= 1.0f) {
            obj.trajectoryT -= 1.0f;
            obj.currentTrajectoryIndex = (obj.currentTrajectoryIndex + 1) % numSegments;
        }

        int base = obj.currentTrajectoryIndex * 4;
        obj.translation = bezierCubic(
            obj.trajectoryPoints[base + 0],
            obj.trajectoryPoints[base + 1],
            obj.trajectoryPoints[base + 2],
            obj.trajectoryPoints[base + 3],
            obj.trajectoryT
        );
        return;
    }

    if (obj.currentTrajectoryIndex < 0 || obj.currentTrajectoryIndex >= static_cast<int>(obj.trajectoryPoints.size())) {
        obj.currentTrajectoryIndex = 0;
    }

    const glm::vec3 target = obj.trajectoryPoints[obj.currentTrajectoryIndex];
    glm::vec3 direction = target - obj.translation;
    float distance = glm::length(direction);
    if (distance < 0.01f) {
        obj.currentTrajectoryIndex = (obj.currentTrajectoryIndex + 1) % static_cast<int>(obj.trajectoryPoints.size());
        return;
    }

    glm::vec3 movement = glm::normalize(direction) * obj.trajectorySpeed * deltaTime;
    if (glm::length(movement) >= distance) {
        obj.translation = target;
        obj.currentTrajectoryIndex = (obj.currentTrajectoryIndex + 1) % static_cast<int>(obj.trajectoryPoints.size());
    } else {
        obj.translation += movement;
    }
}

bool loadMTL(const string& filePath, unordered_map<string, MaterialInfo>& materialMap)
{
    ifstream file(filePath);
    if (!file.is_open()) {
        cerr << "Erro ao abrir arquivo MTL: " << filePath << endl;
        return false;
    }

    auto trim = [](string value) {
        size_t start = value.find_first_not_of(" \t\r\n");
        if (start == string::npos) {
            return string();
        }
        size_t end = value.find_last_not_of(" \t\r\n");
        return value.substr(start, end - start + 1);
    };

    string currentMaterial;
    string line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        istringstream ss(line);
        string word;
        ss >> word;
        if (word == "newmtl") {
            ss >> currentMaterial;
            if (!currentMaterial.empty()) {
                materialMap[currentMaterial] = MaterialInfo();
            }
        } else if (!currentMaterial.empty()) {
            if (word == "Ka") {
                ss >> materialMap[currentMaterial].Ka.r >> materialMap[currentMaterial].Ka.g >> materialMap[currentMaterial].Ka.b;
            } else if (word == "Kd") {
                ss >> materialMap[currentMaterial].Kd.r >> materialMap[currentMaterial].Kd.g >> materialMap[currentMaterial].Kd.b;
            } else if (word == "Ks") {
                ss >> materialMap[currentMaterial].Ks.r >> materialMap[currentMaterial].Ks.g >> materialMap[currentMaterial].Ks.b;
            } else if (word == "Ns") {
                ss >> materialMap[currentMaterial].Ns;
            } else if (word == "map_Kd") {
                string value;
                getline(ss, value);
                materialMap[currentMaterial].diffuseTexture = trim(value);
            } else if (word == "map_d") {
                string value;
                getline(ss, value);
                materialMap[currentMaterial].opacityTexture = trim(value);
            }
        }
    }

    file.close();
    return true;
}

bool loadSceneConfig(const string& filePath)
{
    ifstream file(filePath);
    if (!file.is_open()) {
        cerr << "Nao foi possivel abrir arquivo de cena: " << filePath << endl;
        return false;
    }

    string line;
    string sceneData;
    while (getline(file, line)) {
        sceneData += line;
    }
    file.close();

    auto getString = [&](const string& key, const string& text, const string& defaultValue = "") {
        size_t pos = text.find('"' + key + '"');
        if (pos == string::npos) return defaultValue;
        pos = text.find(':', pos);
        if (pos == string::npos) return defaultValue;
        pos = text.find('"', pos);
        if (pos == string::npos) return defaultValue;
        size_t end = text.find('"', pos + 1);
        if (end == string::npos) return defaultValue;
        return text.substr(pos + 1, end - pos - 1);
    };

    auto getFloat = [&](const string& key, const string& text, float defaultValue = 0.0f) {
        size_t pos = text.find('"' + key + '"');
        if (pos == string::npos) return defaultValue;
        pos = text.find(':', pos);
        if (pos == string::npos) return defaultValue;
        size_t start = text.find_first_not_of(" \t\n\r", pos + 1);
        if (start == string::npos) return defaultValue;
        size_t end = text.find_first_of(",}]\n\r", start);
        string token = text.substr(start, end - start);
        return static_cast<float>(atof(token.c_str()));
    };

    auto getVec3 = [&](const string& key, const string& text, const glm::vec3& defaultValue) {
        size_t pos = text.find('"' + key + '"');
        if (pos == string::npos) return defaultValue;
        pos = text.find('[', pos);
        if (pos == string::npos) return defaultValue;
        size_t end = text.find(']', pos);
        if (end == string::npos) return defaultValue;
        string token = text.substr(pos + 1, end - pos - 1);
        glm::vec3 result = defaultValue;
        sscanf(token.c_str(), "%f,%f,%f", &result.x, &result.y, &result.z);
        return result;
    };

    auto getMatchingBracket = [&](const string& text, size_t openPos, char openChar, char closeChar) {
        if (openPos == string::npos || openPos >= text.size() || text[openPos] != openChar) {
            return string::npos;
        }

        int depth = 0;
        for (size_t i = openPos; i < text.size(); ++i) {
            if (text[i] == openChar) {
                ++depth;
            } else if (text[i] == closeChar) {
                --depth;
                if (depth == 0) {
                    return i;
                }
            }
        }
        return string::npos;
    };

    // Parse camera.
    configuredCameraPosition = getVec3("cameraPosition", sceneData, glm::vec3(0.0f, 0.0f, 5.0f));
    configuredCameraYaw = getFloat("cameraYaw", sceneData, -90.0f);
    configuredCameraPitch = getFloat("cameraPitch", sceneData, 0.0f);
    configuredCameraFov = getFloat("cameraFov", sceneData, 45.0f);
    configuredCameraNearPlane = getFloat("cameraNear", sceneData, 0.1f);
    configuredCameraFarPlane = getFloat("cameraFar", sceneData, 100.0f);
    configuredBackgroundColor = getVec3("backgroundColor", sceneData, glm::vec3(0.95f, 0.95f, 0.95f));
    camera = Camera(configuredCameraPosition, configuredCameraYaw, configuredCameraPitch);

    // Parse luzes.
    for (int i = 0; i < NUM_LIGHTS; i++) {
        string lightKey = "light" + to_string(i + 1);
        sceneLights[i].position = getVec3(lightKey + "Position", sceneData, sceneLights[i].position);
        sceneLights[i].color = getVec3(lightKey + "Color", sceneData, sceneLights[i].color);
        sceneLights[i].intensity = getFloat(lightKey + "Intensity", sceneData, sceneLights[i].intensity);
        sceneLights[i].enabled = getFloat(lightKey + "Enabled", sceneData, sceneLights[i].enabled ? 1.0f : 0.0f) > 0.5f;
    }

    // Parse objects array.
    size_t objectsArrayPos = sceneData.find("\"objects\"");
    if (objectsArrayPos == string::npos) {
        cerr << "Arquivo de cena não contém array 'objects'." << endl;
        return false;
    }

    size_t arrayStart = sceneData.find('[', objectsArrayPos);
    if (arrayStart == string::npos) {
        cerr << "Array 'objects' inválido no arquivo de cena." << endl;
        return false;
    }
    size_t arrayEnd = getMatchingBracket(sceneData, arrayStart, '[', ']');
    if (arrayEnd == string::npos) {
        cerr << "Array 'objects' não fechado no arquivo de cena." << endl;
        return false;
    }

    string objectsText = sceneData.substr(arrayStart + 1, arrayEnd - arrayStart - 1);
    size_t pos = 0;
    while (pos < objectsText.size()) {
        size_t objStart = objectsText.find('{', pos);
        if (objStart == string::npos) {
            break;
        }
        size_t objEnd = getMatchingBracket(objectsText, objStart, '{', '}');
        if (objEnd == string::npos) {
            cerr << "Objeto inválido no array 'objects'." << endl;
            return false;
        }

        string objectText = objectsText.substr(objStart, objEnd - objStart + 1);
        if (objectText.find("\"file\"") == string::npos) {
            pos = objEnd + 1;
            continue;
        }

        ObjModel obj;
        obj.name = getString("name", objectText, getString("file", objectText, ""));
        string fileName = getString("file", objectText, "");
        obj.translation = getVec3("translation", objectText, glm::vec3(0.0f));
        obj.rotation = getVec3("rotation", objectText, glm::vec3(0.0f));
        obj.scale = getVec3("scale", objectText, glm::vec3(1.0f));
        obj.color = getVec3("color", objectText, glm::vec3(0.8f));
        obj.trajectorySpeed  = getFloat("trajectorySpeed",   objectText, 1.2f);
        obj.useBezier        = getFloat("useBezier",         objectText, 0.0f) > 0.5f;
        obj.followTrajectory = getFloat("followTrajectory",  objectText, 0.0f) > 0.5f;

        size_t trajectoryPos = objectText.find("\"trajectory\"");
        if (trajectoryPos != string::npos) {
            size_t trajectoryArrayStart = objectText.find('[', trajectoryPos);
            size_t trajectoryArrayEnd = getMatchingBracket(objectText, trajectoryArrayStart, '[', ']');
            if (trajectoryArrayStart != string::npos && trajectoryArrayEnd != string::npos) {
                string trajectoryText = objectText.substr(trajectoryArrayStart + 1, trajectoryArrayEnd - trajectoryArrayStart - 1);
                size_t trajectoryCursor = 0;
                while (trajectoryCursor < trajectoryText.size()) {
                    size_t pointStart = trajectoryText.find('[', trajectoryCursor);
                    if (pointStart == string::npos) {
                        break;
                    }
                    size_t pointEnd = getMatchingBracket(trajectoryText, pointStart, '[', ']');
                    if (pointEnd == string::npos) {
                        break;
                    }

                    string pointText = trajectoryText.substr(pointStart + 1, pointEnd - pointStart - 1);
                    glm::vec3 trajectoryPoint(0.0f);
                    sscanf(pointText.c_str(), "%f,%f,%f", &trajectoryPoint.x, &trajectoryPoint.y, &trajectoryPoint.z);
                    obj.trajectoryPoints.push_back(trajectoryPoint);
                    trajectoryCursor = pointEnd + 1;
                }
            }
        }

        if (!loadSimpleOBJ(fileName, obj.subMeshes, obj.color)) {
            cerr << "Falha ao carregar objeto: " << fileName << endl;
        } else {
            sceneObjects.push_back(obj);
        }

        pos = objEnd + 1;
    }

    return !sceneObjects.empty();
}

// Esta função `loadTexture` é responsável por carregar uma imagem de um arquivo e criar uma textura OpenGL a partir dela, retornando o identificador da textura (GLuint) para uso posterior na aplicação. A função utiliza a biblioteca stb_image para ler os dados da imagem e configurar os parâmetros da textura no OpenGL.
GLuint loadTexture(const string& filePath)
{
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* data = stbi_load(filePath.c_str(), &width, &height, &channels, 0);
    if (!data) {
        cerr << "Erro ao carregar textura: " << filePath << endl;
        return 0;
    }

    GLenum format = GL_RGB;
    if (channels == 1) format = GL_RED;
    else if (channels == 3) format = GL_RGB;
    else if (channels == 4) format = GL_RGBA;

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    stbi_image_free(data);
    return textureID;
}

GLuint loadTextureWithAlphaMask(const string& diffusePath, const string& alphaPath)
{
    int diffuseWidth = 0;
    int diffuseHeight = 0;
    int diffuseChannels = 0;
    unsigned char* diffuseData = stbi_load(diffusePath.c_str(), &diffuseWidth, &diffuseHeight, &diffuseChannels, 0);
    if (!diffuseData) {
        cerr << "Erro ao carregar textura difusa: " << diffusePath << endl;
        return 0;
    }

    int alphaWidth = 0;
    int alphaHeight = 0;
    int alphaChannels = 0;
    unsigned char* alphaData = stbi_load(alphaPath.c_str(), &alphaWidth, &alphaHeight, &alphaChannels, 0);
    if (!alphaData) {
        cerr << "Erro ao carregar mascara de opacidade: " << alphaPath << endl;
        stbi_image_free(diffuseData);
        return 0;
    }

    if (diffuseWidth != alphaWidth || diffuseHeight != alphaHeight) {
        cerr << "Textura difusa e mascara de opacidade com dimensoes diferentes: " << diffusePath << " / " << alphaPath << endl;
        stbi_image_free(diffuseData);
        stbi_image_free(alphaData);
        return 0;
    }

    vector<unsigned char> rgba(static_cast<size_t>(diffuseWidth) * diffuseHeight * 4);
    for (int y = 0; y < diffuseHeight; ++y) {
        for (int x = 0; x < diffuseWidth; ++x) {
            size_t diffuseIndex = (static_cast<size_t>(y) * diffuseWidth + x) * diffuseChannels;
            size_t alphaIndex = (static_cast<size_t>(y) * diffuseWidth + x) * alphaChannels;
            size_t outIndex = (static_cast<size_t>(y) * diffuseWidth + x) * 4;

            rgba[outIndex + 0] = diffuseData[diffuseIndex + 0];
            rgba[outIndex + 1] = diffuseChannels > 1 ? diffuseData[diffuseIndex + 1] : diffuseData[diffuseIndex + 0];
            rgba[outIndex + 2] = diffuseChannels > 2 ? diffuseData[diffuseIndex + 2] : diffuseData[diffuseIndex + 0];
            unsigned char alphaValue = alphaData[alphaIndex + 0];
            if (alphaChannels > 1) {
                alphaValue = alphaData[alphaIndex + (alphaChannels - 1)];
            }
            rgba[outIndex + 3] = alphaValue;
        }
    }

    stbi_image_free(diffuseData);
    stbi_image_free(alphaData);

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, diffuseWidth, diffuseHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    return textureID;
}

// Esta função carrega arquivos OBJ e cria submeshes separados por material, aplicando .mtl e texturas.
bool loadSimpleOBJ(const string& filePATH, vector<Mesh>& outSubMeshes, const glm::vec3& defaultColor)
{
    vector<glm::vec3> vertices;
    vector<glm::vec2> texCoords;
    vector<glm::vec3> normals;
    unordered_map<string, MaterialInfo> materialMap;
    string currentMaterialName;
    string directory = fs::path(filePATH).parent_path().string();
    string line;
    vector<GLfloat> activeBuffer;
    vector<GLfloat> allBuffers;
    vector<int> materialStartIndices;
    vector<string> materialNames;
    string currentMtlFile;

    struct FaceVertex {
        int vi = 0;
        int ti = 0;
        int ni = 0;
    };

    auto resolveIndex = [](int index, int size) {
        if (size <= 0 || index == 0) {
            return -1;
        }
        if (index > 0) {
            return index - 1;
        }
        int resolved = size + index;
        return (resolved >= 0 && resolved < size) ? resolved : -1;
    };

    auto appendVertex = [&](const FaceVertex& fv) {
        int vi = resolveIndex(fv.vi, static_cast<int>(vertices.size()));
        int ti = resolveIndex(fv.ti, static_cast<int>(texCoords.size()));
        int ni = resolveIndex(fv.ni, static_cast<int>(normals.size()));

        glm::vec3 position = (vi >= 0 && vi < static_cast<int>(vertices.size())) ? vertices[vi] : glm::vec3(0.0f);
        glm::vec2 texCoord = (ti >= 0 && ti < static_cast<int>(texCoords.size())) ? texCoords[ti] : glm::vec2(0.0f);
        glm::vec3 normal = (ni >= 0 && ni < static_cast<int>(normals.size())) ? normals[ni] : glm::vec3(0.0f, 0.0f, 1.0f);

        activeBuffer.push_back(position.x);
        activeBuffer.push_back(position.y);
        activeBuffer.push_back(position.z);
        activeBuffer.push_back(defaultColor.r);
        activeBuffer.push_back(defaultColor.g);
        activeBuffer.push_back(defaultColor.b);
        activeBuffer.push_back(texCoord.x);
        activeBuffer.push_back(texCoord.y);
        activeBuffer.push_back(normal.x);
        activeBuffer.push_back(normal.y);
        activeBuffer.push_back(normal.z);
    };

    ifstream arqEntrada(filePATH.c_str());
    if (!arqEntrada.is_open()) {
        cerr << "Erro ao tentar ler o arquivo " << filePATH << endl;
        return false;
    }

    while (getline(arqEntrada, line)) {
        istringstream ssline(line);
        string word;
        ssline >> word;

        if (word == "v") {
            glm::vec3 vertice;
            ssline >> vertice.x >> vertice.y >> vertice.z;
            vertices.push_back(vertice);
        } else if (word == "vt") {
            glm::vec2 vt;
            ssline >> vt.s >> vt.t;
            texCoords.push_back(vt);
        } else if (word == "vn") {
            glm::vec3 normal;
            ssline >> normal.x >> normal.y >> normal.z;
            normals.push_back(normal);
        } else if (word == "mtllib") {
            ssline >> currentMtlFile;
            string mtlPath = (fs::path(directory) / currentMtlFile).string();
            loadMTL(mtlPath, materialMap);
        } else if (word == "usemtl") {
            string newMaterialName;
            ssline >> newMaterialName;
            if (!activeBuffer.empty()) {
                materialStartIndices.push_back(static_cast<int>(allBuffers.size() / 11));
                materialNames.push_back(currentMaterialName);
                allBuffers.insert(allBuffers.end(), activeBuffer.begin(), activeBuffer.end());
                activeBuffer.clear();
            }
            currentMaterialName = newMaterialName;
        } else if (word == "f") {
            vector<FaceVertex> faceVertices;
            while (ssline >> word) {
                FaceVertex fv;
                istringstream ss(word);
                string index;
                if (getline(ss, index, '/')) fv.vi = !index.empty() ? stoi(index) : 0;
                if (getline(ss, index, '/')) fv.ti = !index.empty() ? stoi(index) : 0;
                if (getline(ss, index)) fv.ni = !index.empty() ? stoi(index) : 0;
                faceVertices.push_back(fv);
            }

            if (faceVertices.size() < 3) {
                continue;
            }

            for (size_t i = 1; i + 1 < faceVertices.size(); ++i) {
                appendVertex(faceVertices[0]);
                appendVertex(faceVertices[i]);
                appendVertex(faceVertices[i + 1]);
            }
        }
    }

    if (!activeBuffer.empty()) {
        materialStartIndices.push_back(static_cast<int>(allBuffers.size() / 11));
        materialNames.push_back(currentMaterialName);
        allBuffers.insert(allBuffers.end(), activeBuffer.begin(), activeBuffer.end());
    }
    arqEntrada.close();

    if (allBuffers.empty()) {
        return false;
    }

    for (size_t i = 0; i < materialNames.size(); ++i) {
        int startVertex = materialStartIndices[i];
        int endVertex = (i + 1 < materialStartIndices.size()) ? materialStartIndices[i + 1] : static_cast<int>(allBuffers.size() / 11);
        int count = endVertex - startVertex;
        if (count <= 0) continue;

        Mesh submesh;
        vector<GLfloat> bufferSegment(allBuffers.begin() + startVertex * 11, allBuffers.begin() + endVertex * 11);
        glGenBuffers(1, &submesh.VBO);
        glBindBuffer(GL_ARRAY_BUFFER, submesh.VBO);
        glBufferData(GL_ARRAY_BUFFER, bufferSegment.size() * sizeof(GLfloat), bufferSegment.data(), GL_STATIC_DRAW);

        glGenVertexArrays(1, &submesh.VAO);
        glBindVertexArray(submesh.VAO);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (GLvoid*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (GLvoid*)(8 * sizeof(GLfloat)));
        glEnableVertexAttribArray(3);
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        submesh.vertexCount = count;
        submesh.baseColor = defaultColor;
        if (!materialNames[i].empty() && materialMap.count(materialNames[i]) > 0) {
            submesh.material = materialMap[materialNames[i]];
            if (!submesh.material.diffuseTexture.empty()) {
                string texturePath = (fs::path(directory) / submesh.material.diffuseTexture).string();
                if (!submesh.material.opacityTexture.empty()) {
                    string alphaPath = (fs::path(directory) / submesh.material.opacityTexture).string();
                    submesh.textureID = loadTextureWithAlphaMask(texturePath, alphaPath);
                } else {
                    submesh.textureID = loadTexture(texturePath);
                }
                submesh.textured = submesh.textureID != 0;
            }
            if (!submesh.textured) {
                submesh.baseColor = submesh.material.Kd;
            }
        }
        outSubMeshes.push_back(submesh);
    }

    return !outSubMeshes.empty();
}
