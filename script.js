// coloque isso dentro da pasta static
// Função para mostrar o spinner
function showSpinner() {
  document.getElementById("loading-spinner").style.display = "block";
}

// Função para esconder o spinner
function hideSpinner() {
  document.getElementById("loading-spinner").style.display = "none";
}

// Quando o formulário for enviado
document.getElementById("question-form").addEventListener("submit", function(event) {
  event.preventDefault();

  // Obtenha a pergunta do campo de texto
  var question = document.getElementById("question").value;

  // Mostre o spinner
  showSpinner();

  // Faça uma solicitação AJAX para o servidor (mantendo o código AJAX anterior)
  $.ajax({
    type: "POST",
    url: "/ask",
    data: { question: question }, // Envie a pergunta para o servidor
    dataType: 'json', // Defina o tipo de dados esperado como JSON
    success: function(response) {
      // Atualize a div de resposta com a resposta do servidor
      document.getElementById("response").innerText = response.response;
    },
    error: function() {
      // Em caso de erro, exiba uma mensagem
      document.getElementById("response").innerText = "Erro ao processar a pergunta.";
    },
    complete: function() {
      // Oculte o spinner quando a solicitação estiver completa
      hideSpinner();
    }
  });
});
