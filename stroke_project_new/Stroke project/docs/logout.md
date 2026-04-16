Logout implemented

Where it's implemented:

- View: `accounts/views.py` -> `UserLogoutView` (uses `django.contrib.auth.logout`) — logs out and redirects to `accounts:login` with a success message.
- URL: `accounts/urls.py` -> path `'logout/'`, name `'logout'`.
- Template: Navbar in `templates/base.html` contains a logout link pointing to `{% url 'accounts:logout' %}` inside the user dropdown.

How to test manually:

1. Start the dev server:

```bash
python manage.py runserver
```

2. Login via the app (or go to `/accounts/login/`).
3. Click the user menu in the navbar and choose "Logout"; you should be redirected to the login page with a success message.

Direct URL: `http://127.0.0.1:8000/accounts/logout/` (GET will log out the current user).

Notes:
- No additional template is required because the logout view redirects to the login page.
- If you want a confirmation page before logout, I can add a `logout_confirm.html` and change the view to render it on GET and perform logout on POST.
