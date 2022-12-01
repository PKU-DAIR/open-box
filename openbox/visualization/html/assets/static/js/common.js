var owner = $.cookie('owner');
function check_login() {
    if (owner == '' || typeof (owner) == 'undefined') {
        window.location.href = '/user_board/'
    }

}
function isAvailableEmail(sEmail) {
      var re = /^\w+(\.\w+)*@\w+\.\w+(\.\w+)*$/i;
      return re.test(sEmail);
    }
